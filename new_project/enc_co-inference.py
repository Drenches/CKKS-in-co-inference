import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import tenseal as ts
from utils import *
import pdb
import datetime
## Load data and model
trained_model = ConvNet()
trained_model.load_state_dict(torch.load('new_project/trained_model_parameters.pkl'))
server_model = ServerNet(trained_model)
user_model = UserNet(trained_model)

test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
testiter = iter(test_loader)

# prepare for cnov
kernel_shape = trained_model.conv1.kernel_size
stride = trained_model.conv1.stride[0]

## Encryption Parameters

# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
starttime = datetime.datetime.now()
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

SingeTest = True
if SingeTest:
    round = 1
    for i in range(round):
        data, target = testiter.next()
        x_enc, windows_nb = GenCiph(data, context, kernel_shape, stride)
        enc_feature_map = ServerInference(server_model, x_enc, windows_nb, kernel_shape, stride)
        pdb.set_trace()
        feature_map = enc_feature_map.decrypt()
        feature_map = torch.tensor(feature_map)
        output = user_model(feature_map)
        pred = torch.argmax(output)
        print(pred)
    endtime = datetime.datetime.now()
else:
    round = 10000
    criterion = torch.nn.CrossEntropyLoss()
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for i in range(round):
        data, target = testiter.next()
        x_enc, windows_nb = GenCiph(data, context, kernel_shape, stride)
        enc_feature_map = ServerInference(server_model, x_enc, windows_nb, kernel_shape, stride)
        feature_map = enc_feature_map.decrypt()
        feature_map = torch.tensor(feature_map)
        output = user_model(feature_map)
        output = torch.tensor(output).view(1, -1)
        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )
    endtime = datetime.datetime.now()
print (endtime - starttime)


    