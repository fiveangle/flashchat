#ifndef FLASHCHAT_SERVER_TITLE_H
#define FLASHCHAT_SERVER_TITLE_H
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>

static NSString *server_title_text(id content) {
    if (!content || content == [NSNull null]) return @"";
    if ([content isKindOfClass:[NSString class]]) return content;
    if ([content isKindOfClass:[NSDictionary class]])
        return [content[@"text"] isKindOfClass:[NSString class]] ? content[@"text"] : nil;
    if (![content isKindOfClass:[NSArray class]]) return nil;
    NSMutableString *text = [NSMutableString string];
    for (id part in content) {
        if ([part isKindOfClass:[NSString class]]) {
            [text appendString:part];
            continue;
        }
        if (![part isKindOfClass:[NSDictionary class]]) return nil;
        id value = part[@"text"] ?: part[@"content"] ?: part[@"input"];
        if (![value isKindOfClass:[NSString class]]) return nil;
        if (text.length) [text appendString:@"\n"];
        [text appendString:value];
    }
    return text;
}

static NSString *server_title_json(NSDictionary *value) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:value options:0 error:NULL];
    return data ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
}

static NSString *server_title_from_messages(NSArray *messages) {
    NSCharacterSet *whitespace = [NSCharacterSet whitespaceAndNewlineCharacterSet];
    for (id message in messages) {
        if (![message isKindOfClass:[NSDictionary class]] ||
            ![message[@"role"] isEqual:@"user"]) continue;
        id content = message[@"content"];
        NSString *text;
        if ([content isKindOfClass:[NSArray class]]) {
            NSMutableArray *parts = [NSMutableArray array];
            for (id part in content) {
                NSString *part_text = server_title_text(part);
                if (part_text.length) [parts addObject:part_text];
            }
            text = [parts componentsJoinedByString:@" "];
        } else {
            text = server_title_text(content);
        }
        if (!text) continue;
        NSMutableArray *words = [NSMutableArray array];
        for (NSString *word in [text componentsSeparatedByCharactersInSet:whitespace]) {
            if (word.length) [words addObject:word];
        }
        text = [words componentsJoinedByString:@" "];
        // OpenCode sends this wrapper as a separate user message.
        if (!text.length || [text isEqualToString:@"Generate a title for this conversation:"]) continue;

        // Count visible characters so truncation cannot split emoji or accents.
        __block NSUInteger count = 0, prefix_end = 0;
        [text enumerateSubstringsInRange:NSMakeRange(0, text.length)
                                options:NSStringEnumerationByComposedCharacterSequences |
                                        NSStringEnumerationSubstringNotRequired
                             usingBlock:^(NSString *substring, NSRange range, NSRange enclosing, BOOL *stop) {
            (void)substring; (void)enclosing;
            if (++count <= 49) prefix_end = NSMaxRange(range);
            if (count > 50) *stop = YES;
        }];
        if (count <= 50) return text;
        NSString *prefix = [text substringToIndex:prefix_end];
        if (![prefix hasSuffix:@" "] && [text characterAtIndex:prefix_end] != ' ') {
            NSRange boundary = [prefix rangeOfString:@" " options:NSBackwardsSearch];
            if (boundary.location != NSNotFound) prefix = [prefix substringToIndex:boundary.location];
        }
        return [[prefix stringByTrimmingCharactersInSet:whitespace] stringByAppendingString:@"…"];
    }
    return @"New conversation";
}

// The OpenCode title uses no inference state and must not reserve its slot.
// Return a bounded, owned HTTP response for the event thread's nonblocking writer.
static char *server_title_response(const char *path, const char *body, const char *model_id) {
    if (strcmp(path, "/v1/chat/completions")) return NULL;
    @autoreleasepool {
        NSData *data = [NSData dataWithBytes:body length:strlen(body)];
        id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
        if (![root isKindOfClass:[NSDictionary class]]) return NULL;
        id messages = root[@"messages"];
        if (![messages isKindOfClass:[NSArray class]]) return NULL;
        NSString *system = nil;
        for (id message in messages) {
            if (![message isKindOfClass:[NSDictionary class]]) return NULL;
            id role = message[@"role"];
            if (![role isKindOfClass:[NSString class]]) return NULL;
            if (![role isEqualToString:@"system"] && ![role isEqualToString:@"developer"]) continue;
            NSString *text = server_title_text(message[@"content"]);
            if (!text) return NULL;
            text = [text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            if (text.length) { system = text; break; }
        }
        if (![system hasPrefix:@"You are a title generator"]) return NULL;
        NSString *title = server_title_from_messages(messages);
        id stream = root[@"stream"];
        if (stream && ![stream isKindOfClass:[NSNumber class]]) return NULL;
        id model = root[@"model"] ?: [NSString stringWithUTF8String:model_id];
        if (![model isKindOfClass:[NSString class]] || [model length] > 255) return NULL;
        NSString *request_id = [@"chatcmpl-title-" stringByAppendingString:[[NSUUID UUID] UUIDString]];
        NSMutableDictionary *reply = [@{@"id": request_id, @"model": model,
            @"created": @((long)[[NSDate date] timeIntervalSince1970])} mutableCopy];
        NSString *payload;
        if ([stream boolValue]) {
            reply[@"object"] = @"chat.completion.chunk";
            NSMutableString *events = [NSMutableString string];
            for (NSDictionary *delta in @[@{@"role": @"assistant", @"content": @""},
                                          @{@"content": title}, @{}]) {
                reply[@"choices"] = @[@{@"index": @0, @"delta": delta,
                    @"finish_reason": delta.count ? [NSNull null] : @"stop"}];
                [events appendFormat:@"data: %@\n\n", server_title_json(reply)];
            }
            [events appendString:@"data: [DONE]\n\n"];
            payload = events;
        } else {
            reply[@"object"] = @"chat.completion";
            reply[@"choices"] = @[@{@"index": @0, @"message": @{@"role": @"assistant",
                @"content": title}, @"finish_reason": @"stop"}];
            payload = server_title_json(reply);
        }
        if (!payload) return NULL;
        NSString *response = [NSString stringWithFormat:
            @"HTTP/1.1 200 OK\r\nContent-Type: %@\r\nAccess-Control-Allow-Origin: *\r\n"
             "Cache-Control: no-cache\r\nConnection: close\r\nContent-Length: %zu\r\n\r\n%@",
            [stream boolValue] ? @"text/event-stream" : @"application/json",
            strlen([payload UTF8String]), payload];
        return strdup([response UTF8String]);
    }
}
#endif
