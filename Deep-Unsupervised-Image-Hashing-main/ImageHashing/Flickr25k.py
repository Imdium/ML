import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import torchvision
from cal_map_mult import calculate_top_map, calculate_map, compress
import data.flickr25k as flickr25k
import logging

# Hyper Parameters
num_epochs = 200
batch_size = 32
epoch_lr_decrease = 60
learning_rate = 0.0001
gamma = 6
encode_length = 16
num_classes = 24
save_interval = 10  # Save the model every 10 epochs
checkpoint_dir = './checkpoints'
log_file = 'Flickr25_training_log.txt'

# Bi-half layer
class hash(Function):
    @staticmethod
    def forward(ctx, U):
        # Yunqiang for half and half (optimal transport)
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N/2), D]), -torch.ones([N - int(N/2), D]))).cuda()    
        B = torch.zeros(U.shape).cuda().scatter_(0, index, B_creat)
        
        ctx.save_for_backward(U, B) 
        
        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B)/(B.numel())
        grad = g + gamma*add_g
        return grad

def hash_layer(input):
    return hash.apply(input)

class CNN(nn.Module):
    def __init__(self, encode_length):
        super(CNN, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False
        torch.manual_seed(0)
        self.fc_encode = nn.Linear(4096, encode_length)

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        h = self.fc_encode(x)
        b = hash_layer(h)
        return x, h, b

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
#################        
def save_checkpoint(epoch, model, optimizer, loss, filename):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        return model, optimizer, start_epoch, loss
    else:
        return model, optimizer, 0, None


def main():
    ###
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logging.basicConfig(filename=log_file, level=logging.INFO)
    ####
    test_loader, train_loader, database_loader = flickr25k.load_data(root='/data2/huwentao/gzy/DeepBit/ImageHashing/data/Flickr25k',
                                                                                )

    #cnn = CNN(encode_length=encode_length)
    cnn = CNN(encode_length=encode_length).cuda()

    # Loss and Optimizer
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    #####
    start_epoch = 0
    loss = None
    # Load checkpoint if available
    checkpoint_file = os.path.join(checkpoint_dir, 'Flickr25k_checkpoint.pth')
    cnn, optimizer, start_epoch, loss = load_checkpoint(checkpoint_file, cnn, optimizer)
    #####


    # Train the Model
    for epoch in range(num_epochs):
        cnn.cuda().train()
        adjust_learning_rate(optimizer, epoch)
        for i, (images, labels, index) in enumerate(train_loader):
            ##
            images = Variable(images.cuda())
            labels = Variable(labels.cuda().long())
            #images, labels = images.to(device), labels.to(device)
            ##


            # Forward + Backward + Optimize
            optimizer.zero_grad()
            x, _, b = cnn(images)

            target_b = F.cosine_similarity(b[:int(labels.size(0) / 2)], b[int(labels.size(0) / 2):])
            target_x = F.cosine_similarity(x[:int(labels.size(0) / 2)], x[int(labels.size(0) / 2):])
            loss = F.mse_loss(target_b, target_x)
            loss.backward()
            optimizer.step()
            ##
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
 
    
        # Test the Model
        if (epoch + 1) % 10 == 0:
            ##
            save_checkpoint(epoch, cnn, optimizer, loss, checkpoint_file)
            ##
            cnn.eval()
            retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn, classes=num_classes)

            result_map = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
            print('--------mAP@All: {}--------'.format(result_map))  


if __name__ == '__main__':
    main() 