from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph = model_path +'/raw_graph_def.pb',
              input_binary = True,
              input_checkpoint = last_checkpoint,
              output_node_names = "action",
              output_graph = model_path +'/your_name_graph.bytes' ,
              clear_devices = True, initializer_nodes = "",input_saver = "",
              restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0")