#ifndef TFLITE_TASK_H
#define TFLITE_TASK_H

#ifdef __cplusplus
#include <stdint.h>

extern "C" {
#endif
#define IMAGE_0 0x00000010
#define IMAGE_1 0x00000020
#define RAW_PIXEL_COUNT (120 * 160)
#define Notification(var, mask) ((var & mask) == mask)
#define WIFI_SSID "wdhzw_2.4G"
#define WIFI_PASSWORD "tietheibai"
#define TASK_STARTUP_COUNTDOWN 10
#define TASK_STARTUP_INTERVAL 5
void wifi_connect();
bool tflite_init();
void tflite_task();
int predict_image_from_buffer(int image_number, bool attach_image);
void send_results_to_server(int image_number, const char* result, uint16_t *image_data, size_t image_size); 

#ifdef __cplusplus
}
#endif

#endif // TFLITE_TASK_H
