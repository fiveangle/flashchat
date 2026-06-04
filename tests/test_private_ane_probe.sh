#!/usr/bin/env bash
set -euo pipefail

if [[ "${FLASHCHAT_PRIVATE_ANE:-}" != "1" ]]; then
    echo "SKIP: set FLASHCHAT_PRIVATE_ANE=1 to run the private ANE probe"
    exit 0
fi

if [[ ! -d /System/Library/PrivateFrameworks/AppleNeuralEngine.framework ]]; then
    echo "SKIP: AppleNeuralEngine private framework not found"
    exit 0
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/flashchat-private-ane.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

SRC="${TMP_DIR}/private_ane_probe.m"
BIN="${TMP_DIR}/private_ane_probe"

cat > "${SRC}" <<'EOF'
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;

static double ticks_to_ms(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
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

static NSData *build_identity_weight_blob(int dim) {
    int weight_bytes = dim * dim * 2;
    int total = 128 + weight_bytes;
    uint8_t *blob = (uint8_t *)calloc((size_t)total, 1);
    blob[0] = 1;
    blob[4] = 2;
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = 1;
    *(uint32_t *)(blob + 72) = (uint32_t)weight_bytes;
    *(uint32_t *)(blob + 80) = 128;
    _Float16 *w = (_Float16 *)(blob + 128);
    for (int i = 0; i < dim; i++) {
        w[i * dim + i] = (_Float16)1.0f;
    }
    return [NSData dataWithBytesNoCopy:blob length:(NSUInteger)total freeWhenDone:YES];
}

static NSString *gen_mil(int dim, int spatial, NSString *tag) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}, {\"flashchat-private-ane-probe\", \"%@\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n", tag, dim, spatial, dim, spatial, dim, dim, dim, dim, dim, spatial, dim, spatial];
}

static int run_case(int dim, int spatial, int require_success) {
    @autoreleasepool {
        Class desc_class = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class model_class = NSClassFromString(@"_ANEInMemoryModel");
        Class request_class = NSClassFromString(@"_ANERequest");
        Class surface_class = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!desc_class || !model_class || !request_class || !surface_class) {
            printf("SKIP: private ANE classes unavailable\n");
            return require_success ? 2 : 0;
        }

        NSString *tag = [NSString stringWithFormat:@"identity-%d-%d", dim, spatial];
        NSData *mil = [gen_mil(dim, spatial, tag) dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weights = build_identity_weight_blob(dim);
        NSError *error = nil;
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            desc_class, @selector(modelWithMILText:weights:optionsPlist:),
            mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weights}}, nil);
        if (!desc) {
            printf("dim=%d spatial=%d descriptor failed\n", dim, spatial);
            return require_success ? 3 : 0;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(model_class, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            printf("dim=%d spatial=%d model failed\n", dim, spatial);
            return require_success ? 4 : 0;
        }

        id hex_id = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmp_dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex_id];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmp_dir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[tmp_dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weights writeToFile:[tmp_dir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        if (ok) {
            ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        }
        double compile_load_ms = ticks_to_ms(mach_absolute_time() - t0);
        if (!ok) {
            printf("dim=%d spatial=%d compile/load failed: %s\n",
                dim, spatial, error ? [[error description] UTF8String] : "unknown");
            [fm removeItemAtPath:tmp_dir error:nil];
            return require_success ? 5 : 0;
        }

        size_t count = (size_t)dim * (size_t)spatial;
        size_t bytes = count * sizeof(float);
        IOSurfaceRef input_surface = make_surface(bytes);
        IOSurfaceRef output_surface = make_surface(bytes);
        IOSurfaceLock(input_surface, 0, NULL);
        float *input = (float *)IOSurfaceGetBaseAddress(input_surface);
        for (size_t i = 0; i < count; i++) {
            input[i] = (float)((int)(i % 31) - 15) / 16.0f;
        }
        IOSurfaceUnlock(input_surface, 0, NULL);
        IOSurfaceLock(output_surface, 0, NULL);
        memset(IOSurfaceGetBaseAddress(output_surface), 0, bytes);
        IOSurfaceUnlock(output_surface, 0, NULL);

        id input_obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surface_class, @selector(objectWithIOSurface:), input_surface);
        id output_obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surface_class, @selector(objectWithIOSurface:), output_surface);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            request_class, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[input_obj], @[@0], @[output_obj], @[@0], nil, nil, @0);

        error = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
        if (!ok) {
            printf("dim=%d spatial=%d eval failed: %s\n",
                dim, spatial, error ? [[error description] UTF8String] : "unknown");
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &error);
            CFRelease(input_surface);
            CFRelease(output_surface);
            [fm removeItemAtPath:tmp_dir error:nil];
            return require_success ? 6 : 0;
        }

        IOSurfaceLock(input_surface, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(output_surface, kIOSurfaceLockReadOnly, NULL);
        float *in = (float *)IOSurfaceGetBaseAddress(input_surface);
        float *out = (float *)IOSurfaceGetBaseAddress(output_surface);
        float max_error = 0.0f;
        for (size_t i = 0; i < count; i++) {
            float err = fabsf(in[i] - out[i]);
            if (err > max_error) max_error = err;
        }
        IOSurfaceUnlock(output_surface, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(input_surface, kIOSurfaceLockReadOnly, NULL);

        t0 = mach_absolute_time();
        for (int i = 0; i < 10; i++) {
            ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
        }
        double eval_ms = ticks_to_ms(mach_absolute_time() - t0) / 10.0;
        printf("dim=%d spatial=%d ok max_error=%.6f compile_load=%.2fms eval=%.4fms\n",
            dim, spatial, max_error, compile_load_ms, eval_ms);

        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        CFRelease(input_surface);
        CFRelease(output_surface);
        [fm removeItemAtPath:tmp_dir error:nil];

        if (max_error > 0.01f) {
            return require_success ? 7 : 0;
        }
        return 0;
    }
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!handle) {
            printf("SKIP: AppleNeuralEngine dlopen failed\n");
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

        int rc = run_case(16, 16, 1);
        if (rc != 0) return rc;
        return run_case(16, 8, 0);
    }
}
EOF

clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl "${SRC}" -o "${BIN}"
"${BIN}"
