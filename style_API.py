import sys
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from models import TransformerNet, VGG16
from utils import *


# training API
# params:
# >style_img_path: 风格图像的文件路径（含文件名）
# >style_model_path: 模型的保存地址（含文件名）


def train_new_style(style_img_path, style_model_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # Basic params settings
    dataset_path = "datasets"  # 此处为coco14数据集的地址
    epochs = 1
    batch_size = 4
    # max_train_batch = 20000
    image_size = 256
    style_size = None
    # 以下三个参数值可能需要修改
    # 1. 1e3 1e6 1 ep=24000
    # 2. 1e2 1e5 0.5 ep=18000
    # 3. 5e1 5e4 0.01 ep=max lr=1e-4
    # 原论文lua实现中为1.0,5.0,1e-6
    # tensorflow版本中为7.5(15),100
    lambda_content = float(5e1)
    lambda_style = float(5e4)
    lambda_tv = float(0.01)
    lr = float(1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloader for the training data
    train_dataset = datasets.ImageFolder(
        dataset_path, train_transform(image_size))
    dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # Defines networks
    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    # Define optimizer and loss
    optimizer = Adam(transformer.parameters(), lr)
    l2_loss = torch.nn.MSELoss().to(device)

    # Load style image
    style = style_transform(style_size)(Image.open(style_img_path))
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    # Extract style features
    features_style = vgg(style)
    gram_style = [gram_matrix(y) for y in features_style]

    for epoch in range(epochs):
        # epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()

            images_original = images.to(device)
            images_transformed = transformer(images_original)

            # Extract features
            features_original = vgg(images_original)
            features_transformed = vgg(images_transformed)

            # Compute content loss as MSE between features
            content_size = features_transformed.relu2_2.shape[0]*features_transformed.relu2_2.shape[1] * \
                features_transformed.relu2_2.shape[2] * \
                features_transformed.relu2_2.shape[3]
            content_loss = lambda_content*2 * \
                l2_loss(features_transformed.relu2_2,
                        features_original.relu2_2)
            content_loss /= content_size

            # Compute style loss as MSE between gram matrices
            style_loss = 0
            for ft_y, gm_s in zip(features_transformed, gram_style):
                gm_y = gram_matrix(ft_y)
                gm_size = gm_y.shape[0]*gm_y.shape[1]*gm_y.shape[2]
                style_loss += l2_loss(gm_y,
                                      gm_s[: images.size(0), :, :])/gm_size
            style_loss *= lambda_style*2

            # Compute tv loss
            y_tv = l2_loss(
                images_transformed[:, :, 1:, :], images_transformed[:, :, :image_size-1, :])
            x_tv = l2_loss(
                images_transformed[:, :, :, 1:], images_transformed[:, :, :, :image_size-1])
            tv_loss = lambda_tv*2 * \
                (x_tv/image_size + y_tv/image_size)/batch_size

            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

    # Save trained model
    torch.save(transformer.state_dict(), style_model_path)

# transfer API
# params:
# >usr_img_path: 用户原图像的文件路径（含文件名）
# >style_model_path: 模型的地址（含文件名）
# >new_img_path: 迁移完成后的新图片的文件路径（含文件名）


def transfer_img(usr_img_path, style_model_path, new_img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(style_model_path))
    transformer.eval()

    # Prepare input
    transform = style_transform()
    image_tensor = transform(Image.open(usr_img_path)).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    # Save image
    save_image(stylized_image, new_img_path)
