#include <cstddef>
#include "tflite_task.h"
#include "model_data_v1.h"  // the header file containing your model
#include <mbedtls/base64.h>
#include <stdint.h>
#include <math.h>
#include <iostream>
#include "sys_utilities.h"
#include "json_utilities.h"
#include "vospi.h"
#include "nvs_flash.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_tls.h"
#include "esp_crt_bundle.h"
#include "esp_sntp.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
static const char* TAG = "tflite_task";

static bool interpreter_init = false;

/*
    image buffer for quick storage and release of semaphore, and resized image buffer 
*/
uint16_t *resized_image;
uint16_t *raw_image;
static uint8_t global_max_probability_index = 99;



#define IMAGE_WIDTH 120  // Adjust this to your desired width
#define IMAGE_HEIGHT 160  // Adjust this to your desired height
#define IMAGE_CHANNEL 1  // Adjust this to match the number of color channels in your image (typically 3 for RGB, 1 for grayscale)

//Variable associated with tf model
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
const int TENSOR_ARENA_SIZE_BYTE = 5 * 300 * 1024;
static uint8_t *tensor_arena;
/*
static void event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data);

// Event handler for catching WiFi events
// Defined at a global scope
static void event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        }
}

// Function to connect to WiFi
void wifi_connect() {
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL);

    wifi_config_t wifi_config;
    strncpy((char*)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, WIFI_PASSWORD, sizeof(wifi_config.sta.password));


    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "wifi_init_sta finished.");
    ESP_LOGI(TAG, "connecting to ap SSID:%s password:%s",
             WIFI_SSID, WIFI_PASSWORD);
}
*/
bool tflite_init(){
    //wifi_connect();
    model = tflite::GetModel(model_data_v1);
    raw_image = (uint16_t *)heap_caps_malloc(RAW_PIXEL_COUNT * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
    resized_image = (uint16_t *)heap_caps_malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model provided is of version %d but TFLite used is of version %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE_BYTE, MALLOC_CAP_SPIRAM);

    if (tensor_arena == NULL)
    {
        ESP_LOGE(TAG, "Couldn't allocate memory for Tensorflow Lite, required %d bytes", TENSOR_ARENA_SIZE_BYTE);
        ESP_LOGE(TAG, "Available: %d bytes", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
        return false;
    }
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE_BYTE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Tensor allocation failed");
        return false;
    }
    input = interpreter->input(0);
    ESP_LOGI(TAG, "Tensorflow Lite initiation successful");
    ESP_LOGI(TAG, "Memory available: %d bytes", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "Tensor size: %d bytes", interpreter->arena_used_bytes());
    ESP_LOGI(TAG, "Tensor dimension: %d", interpreter->input(0)->dims->size);
    interpreter_init = true;
    return true;
}
void tflite_task()
{
    if (!interpreter_init)
        {
            ESP_LOGE(TAG, "Abort, the interpreter is not initialised");
            vTaskDelete(NULL);
        }
    else{
         ESP_LOGI(TAG, "TFLite task initialised");
        int countdown = 10;
        while(countdown > 0)
        {
            ESP_LOGI(TAG, "Counting down: %d", countdown);
            vTaskDelay(pdMS_TO_TICKS(1000));
            countdown--;
        }
        while (true)
        {
            uint32_t notification_value = 0;
            // Gathering images once a sampling in the ping-pong buffer is completed.
            if (xTaskNotifyWait(0x00, 0xFFFFFFFF, &notification_value, 0))
            {
                //ESP_LOGI(TAG, "notification_value: %d", notification_value);
                if (Notification(notification_value, IMAGE_0))
                {
                    //ESP_LOGI(TAG, "Ping-pong buffer index 0");
                    predict_image_from_buffer(0, false);
                }
                if (Notification(notification_value, IMAGE_1))
                {
                    //ESP_LOGI(TAG, "Ping-pong buffer index 1");
                    predict_image_from_buffer(1, false);
                }
            }
            vTaskDelay(pdMS_TO_TICKS(TASK_STARTUP_INTERVAL * 1000));
        }
          
    }

}



int predict_image_from_buffer(int image_number, bool attach_image)
{
    unsigned char *base64_image = nullptr;
    size_t base64_obj_len;
    const int POST_DATA_LENGTH = 1024 * 56;

    //ESP_LOGI(TAG, "Start prediction for buffer %d", image_number);

    uint16_t *response_buffer = ((lep_buffer_t *)&rsp_lep_buffer[image_number])->lep_bufferP;

    // Locking the mutex
    //ESP_LOGI(TAG, "Taking mutex for buffer %d", image_number);
    xSemaphoreTake(rsp_lep_buffer[image_number].lep_mutex, portMAX_DELAY);

    //ESP_LOGI(TAG, "Copying data from buffer %d", image_number);
    memcpy(raw_image, response_buffer, RAW_PIXEL_COUNT * sizeof(uint16_t));
    
    //ESP_LOGI(TAG, "Releasing mutex for buffer %d", image_number);
    xSemaphoreGive(rsp_lep_buffer[image_number].lep_mutex);

    resized_image = raw_image;

    float mean = 30239.793f, stddev =282.5083f; 
    ESP_LOGI(TAG, "Normalizing image data for buffer %d", image_number);
    for (int i = 0; i < 120 * 160; i++) {
        input->data.f[i] = (resized_image[i] - mean) / stddev;
    }

    //ESP_LOGI(TAG, "Invoking TensorFlow Lite interpreter for buffer %d", image_number);
    if (interpreter->Invoke() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Cannot make a prediction, Invoke fails for buffer %d", image_number);
    }
    else
    {
        TfLiteTensor *output = nullptr;
        output = interpreter->output(0);

        uint8_t max_probability_index = 0;
        for (int i = 0; i < 2; i++) 
        {
            if (output->data.f[i] > output->data.f[max_probability_index])
            {
                max_probability_index = i;
            }
        }
        
        global_max_probability_index = (uint8_t)max_probability_index;

        const char* human_detection_label_list[2] = {"Human", "No human"};
        const char* result = human_detection_label_list[max_probability_index];
        float confidence;
        if (max_probability_index == 0){
            confidence = output -> data.f[0];
        }
        else{
            confidence = output -> data.f[1];
        }

        mbedtls_base64_encode(base64_image, 0, &base64_obj_len, (uint8_t *)resized_image, RAW_PIXEL_COUNT * sizeof(uint16_t));
        base64_image = (unsigned char *)heap_caps_malloc(base64_obj_len, MALLOC_CAP_SPIRAM);
        if (mbedtls_base64_encode(base64_image, base64_obj_len, &base64_obj_len, (uint8_t *)resized_image, RAW_PIXEL_COUNT * sizeof(uint16_t)) != 0) {
            ESP_LOGE(TAG, "Failed to base64 encode image data");
            free(base64_image);
            return 0;
        }

        // Set up HTTP client configuration
        esp_http_client_config_t config = {};
        config.url = "http://192.168.1.111:5000/postdata";  // replace with your server's address and port
        config.method = HTTP_METHOD_POST;
        config.transport_type = HTTP_TRANSPORT_OVER_TCP;

        // Initialize the HTTP client
        esp_http_client_handle_t client = esp_http_client_init(&config);

        // Prepare the POST data
        char *post_data = (char*)heap_caps_malloc(POST_DATA_LENGTH, MALLOC_CAP_SPIRAM);
        sprintf(post_data, "{\"result\": \"%s\", \"confidence\": \"%f\", \"image_data\": \"%s\"}", result, confidence, base64_image);

        // Set the request headers
        esp_http_client_set_header(client, "Content-Type", "application/json");
        
        // Set the POST field data
        esp_http_client_set_post_field(client, post_data, strlen(post_data));

        // Perform the HTTP request
        esp_err_t err = esp_http_client_perform(client);

        // Check the result of the HTTP request
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "HTTP POST Status = %d, content_length = %d",
            esp_http_client_get_status_code(client),
            esp_http_client_get_content_length(client));
        } else {
            ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
        }

        // Cleanup the HTTP client
        esp_http_client_cleanup(client);
        // Free the base64_image
        free(base64_image);  
        free(post_data);

        ESP_LOGI(TAG, "[%d] Human | No human", image_number);
        ESP_LOGI(TAG, "[%d] %.05f | %.05f", image_number, output->data.f[0], output->data.f[1]);
        ESP_LOGI(TAG, "[%d] Result = [%s]", image_number, human_detection_label_list[max_probability_index]);
    }

    ESP_LOGI(TAG, "Prediction complete for buffer %d", image_number);
    return 1;
}
