# CKKS-in-co-inference

This is the implementation using CKKS in co-inference. The core idea is that user uploads its encrypted data (CKKS) and then server execute first part of the pre-trained DNN to get a encrypted inter. feature map, and sends it back to the user. The user then decrypts it into plaintext, and execute the rest of the net. 
