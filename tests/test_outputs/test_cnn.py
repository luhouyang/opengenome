from opengenome.outputs.cnn import viz
import torch


def test_viz():
    from torchvision.models import vgg16, VGG16_Weights
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model = model.to(DEVICE)
    img_car = "tests/data/car_01.png"
    viz(10, img_car, model, DEVICE)
