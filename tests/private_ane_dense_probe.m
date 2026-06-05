#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;

static double ticks_to_ms(uint64_t ticks) {
    return (double)ticks * g_tb.numer / g_tb.denom / 1e6;
}

static double env_double(const char *name, double fallback) {
    const char *value = getenv(name);
    if (!value || !*value) return fallback;
    char *end = NULL;
    double parsed = strtod(value, &end);
    return end && *end == '\0' ? parsed : fallback;
}

static int env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    if (!value || !*value) return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    return end && *end == '\0' && parsed > 0 ? (int)parsed : fallback;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

static NSDictionary *read_json(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) return nil;
    id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [obj isKindOfClass:[NSDictionary class]] ? obj : nil;
}

static NSString *write_model_files(id model, NSData *mil, NSData *weights) {
    id hex_id = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmp_dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex_id ?: @"flashchat-ane-dense"];
    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm createDirectoryAtPath:[tmp_dir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil]) return nil;
    if (![mil writeToFile:[tmp_dir stringByAppendingPathComponent:@"model.mil"] atomically:YES]) return nil;
    if (![weights writeToFile:[tmp_dir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES]) return nil;
    if (![weights writeToFile:[tmp_dir stringByAppendingPathComponent:@"coremldata.bin"] atomically:YES]) return nil;
    return tmp_dir;
}

static NSData *read_file(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) {
        fprintf(stderr, "ERROR: failed to read %s\n", [path UTF8String]);
    }
    return data;
}

static int run_artifact(NSString *artifact_dir) {
    NSDictionary *manifest = read_json([artifact_dir stringByAppendingPathComponent:@"manifest.json"]);
    if (!manifest) {
        fprintf(stderr, "ERROR: missing or invalid manifest.json\n");
        return 2;
    }

    NSNumber *tokens_num = manifest[@"tokens"];
    NSNumber *input_dim_num = manifest[@"input_dim"];
    NSNumber *output_dim_num = manifest[@"output_dim"];
    NSArray *outputs_meta = manifest[@"outputs"];
    NSString *modelc_name = manifest[@"mlmodelc"];
    NSString *input_name = manifest[@"input_file"];
    NSString *reference_name = manifest[@"reference_file"];
    NSString *tensor = manifest[@"source_tensor"] ?: @"unknown";
    if (!tokens_num || !input_dim_num || !output_dim_num || !outputs_meta || !modelc_name || !input_name || !reference_name) {
        fprintf(stderr, "ERROR: manifest is missing required probe fields\n");
        return 3;
    }

    int tokens = [tokens_num intValue];
    int input_dim = [input_dim_num intValue];
    int output_dim = [output_dim_num intValue];
    NSString *modelc_dir = [artifact_dir stringByAppendingPathComponent:modelc_name];
    NSData *mil = read_file([modelc_dir stringByAppendingPathComponent:@"model.mil"]);
    NSString *weights_path = [[modelc_dir stringByAppendingPathComponent:@"weights"] stringByAppendingPathComponent:@"weight.bin"];
    NSData *weights = read_file(weights_path);
    if (!weights) {
        weights = read_file([modelc_dir stringByAppendingPathComponent:@"coremldata.bin"]);
    }
    NSData *input_data = read_file([artifact_dir stringByAppendingPathComponent:input_name]);
    NSData *reference_data = read_file([artifact_dir stringByAppendingPathComponent:reference_name]);
    if (!mil || !weights || !input_data || !reference_data) return 4;

    size_t input_bytes = (size_t)tokens * (size_t)input_dim * sizeof(float);
    size_t reference_bytes = (size_t)tokens * (size_t)output_dim * sizeof(float);
    if ([input_data length] != input_bytes || [reference_data length] != reference_bytes) {
        fprintf(stderr, "ERROR: input/reference vector sizes do not match manifest\n");
        return 5;
    }

    void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!handle) {
        printf("SKIP: AppleNeuralEngine dlopen failed\n");
        return 0;
    }

    Class desc_class = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class model_class = NSClassFromString(@"_ANEInMemoryModel");
    Class request_class = NSClassFromString(@"_ANERequest");
    Class surface_class = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!desc_class || !model_class || !request_class || !surface_class) {
        printf("SKIP: private ANE classes unavailable\n");
        return 0;
    }

    Class device_info = NSClassFromString(@"_ANEDeviceInfo");
    if (device_info) {
        BOOL has_ane = ((BOOL(*)(Class, SEL))objc_msgSend)(device_info, @selector(hasANE));
        id arch = ((id(*)(Class, SEL))objc_msgSend)(device_info, @selector(aneArchitectureType));
        unsigned long cores = ((unsigned long(*)(Class, SEL))objc_msgSend)(device_info, @selector(numANECores));
        printf("ANE hasANE=%d cores=%lu architecture=%s\n",
               has_ane ? 1 : 0, cores, arch ? [[arch description] UTF8String] : "unknown");
        if (!has_ane) {
            printf("SKIP: ANE unavailable\n");
            return 0;
        }
    }

    NSDictionary *weight_entry = @{@"offset": @0, @"data": weights};
    NSDictionary *weight_map = @{
        @"@model_path/weights/weight.bin": weight_entry,
        @"@model_path/coremldata.bin": weight_entry,
    };
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        desc_class, @selector(modelWithMILText:weights:optionsPlist:), mil, weight_map, nil);
    if (!desc) {
        fprintf(stderr, "ERROR: descriptor creation failed\n");
        return 6;
    }
    id model = ((id(*)(Class, SEL, id))objc_msgSend)(model_class, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) {
        fprintf(stderr, "ERROR: in-memory model creation failed\n");
        return 7;
    }

    NSError *error = nil;
    BOOL cached_before = ((BOOL(*)(id, SEL))objc_msgSend)(model, @selector(compiledModelExists));
    uint64_t t0 = mach_absolute_time();
    BOOL temp_compile = [outputs_meta count] > 1;
    NSString *tmp_model_dir = nil;
    if (temp_compile && !(tmp_model_dir = write_model_files(model, mil, weights))) {
        fprintf(stderr, "ERROR: temp model write failed\n");
        return 8;
    }
    BOOL ok = temp_compile ? 0 : ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
    if (!ok || temp_compile) {
        if (!temp_compile) {
            temp_compile = 1;
            tmp_model_dir = write_model_files(model, mil, weights);
            if (!tmp_model_dir) {
                fprintf(stderr, "ERROR: temp model write failed\n");
                return 8;
            }
        }
        error = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        if (ok) {
            error = nil;
            ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        }
    }
    double load_ms = ticks_to_ms(mach_absolute_time() - t0);
    if (tmp_model_dir) {
        [[NSFileManager defaultManager] removeItemAtPath:tmp_model_dir error:nil];
    }
    if (!ok) {
        fprintf(stderr, "ERROR: load failed: %s\n",
                error ? [[error description] UTF8String] : "unknown");
        return 8;
    }

    IOSurfaceRef input_surface = make_surface(input_bytes);
    IOSurfaceLock(input_surface, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(input_surface), [input_data bytes], input_bytes);
    IOSurfaceUnlock(input_surface, 0, NULL);

    NSMutableArray *surface_refs = [NSMutableArray array];
    NSMutableArray *output_objects = [NSMutableArray array];
    NSMutableArray *output_indices = [NSMutableArray array];
    for (NSUInteger i = 0; i < [outputs_meta count]; i++) {
        NSDictionary *entry = outputs_meta[i];
        int dim = [entry[@"dim"] intValue];
        size_t bytes = (size_t)tokens * (size_t)dim * sizeof(float);
        IOSurfaceRef surface = make_surface(bytes);
        IOSurfaceLock(surface, 0, NULL);
        memset(IOSurfaceGetBaseAddress(surface), 0, bytes);
        IOSurfaceUnlock(surface, 0, NULL);
        id obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surface_class, @selector(objectWithIOSurface:), surface);
        [surface_refs addObject:[NSValue valueWithPointer:surface]];
        [output_objects addObject:obj];
        [output_indices addObject:@(i)];
    }

    id input_obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        surface_class, @selector(objectWithIOSurface:), input_surface);
    id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        request_class, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[input_obj], @[@0], output_objects, output_indices, nil, nil, @0);

    error = nil;
    ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
    if (!ok) {
        fprintf(stderr, "ERROR: eval failed: %s\n", error ? [[error description] UTF8String] : "unknown");
        return 9;
    }

    const float *reference = (const float *)[reference_data bytes];
    double max_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    size_t count = 0;
    for (NSUInteger i = 0; i < [outputs_meta count]; i++) {
        NSDictionary *entry = outputs_meta[i];
        int start = [entry[@"start"] intValue];
        int dim = [entry[@"dim"] intValue];
        IOSurfaceRef surface = (IOSurfaceRef)[surface_refs[i] pointerValue];
        IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
        const float *got = (const float *)IOSurfaceGetBaseAddress(surface);
        for (int token = 0; token < tokens; token++) {
            const float *ref_row = reference + (size_t)token * output_dim + start;
            const float *got_row = got + (size_t)token * dim;
            for (int j = 0; j < dim; j++) {
                double diff = (double)got_row[j] - (double)ref_row[j];
                double abs_diff = fabs(diff);
                if (abs_diff > max_abs) max_abs = abs_diff;
                sum_sq += diff * diff;
                ref_sq += (double)ref_row[j] * (double)ref_row[j];
                count++;
            }
        }
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    }

    int repeats = env_int("FLASHCHAT_ANE_DENSE_REPEATS", 20);
    t0 = mach_absolute_time();
    for (int i = 0; i < repeats; i++) {
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
        if (!ok) break;
    }
    double eval_ms = ticks_to_ms(mach_absolute_time() - t0) / repeats;

    double rmse = sqrt(sum_sq / (double)count);
    double rel_rmse = sqrt(sum_sq / (ref_sq > 1e-12 ? ref_sq : 1.0));
    printf("ANE dense projection tensor=%s shape=[%d,%d]->%d outputs=%lu cached=%d temp=%d load=%.2fms eval=%.4fms max_abs=%.6f rmse=%.6f rel_rmse=%.6f\n",
           [tensor UTF8String], tokens, input_dim, output_dim,
           (unsigned long)[outputs_meta count], cached_before ? 1 : 0,
           temp_compile ? 1 : 0, load_ms, eval_ms, max_abs, rmse, rel_rmse);

    double max_abs_tol = env_double("FLASHCHAT_ANE_DENSE_MAX_ABS_TOL", 1.0);
    double rel_rmse_tol = env_double("FLASHCHAT_ANE_DENSE_REL_RMSE_TOL", 0.02);
    if (max_abs > max_abs_tol || rel_rmse > rel_rmse_tol) {
        fprintf(stderr, "ERROR: validation exceeded tolerance max_abs<=%.6f rel_rmse<=%.6f\n",
                max_abs_tol, rel_rmse_tol);
        return 10;
    }

    ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        model, @selector(unloadWithQoS:error:), 21, &error);
    CFRelease(input_surface);
    for (NSValue *value in surface_refs) {
        CFRelease((IOSurfaceRef)[value pointerValue]);
    }
    return ok ? 0 : 11;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        if (argc != 2) {
            fprintf(stderr, "usage: %s ARTIFACT_DIR\n", argv[0]);
            return 1;
        }
        return run_artifact([NSString stringWithUTF8String:argv[1]]);
    }
}
