import torch

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.') #There are 1 GPU(s) available.
    device = torch.device("cpu") #We will use the GPU: GeForce RTX 3090
    
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
