set(SOURCES main.c ctrl_task.c lep_task.c mon_task.c net_cmd_task.c rsp_task.c sif_cmd_task.c tflite_task.cc model_data_v1.cc)
idf_component_register(SRCS ${SOURCES}
                    INCLUDE_DIRS . image_utils "../components"
                    REQUIRES clock cmd i2c lepton sys tflite-lib)

target_compile_definitions(${COMPONENT_LIB} PRIVATE LV_CONF_INCLUDE_SIMPLE=1)