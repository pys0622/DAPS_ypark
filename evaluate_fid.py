import torch
from data import ImageDataset
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from piq import FID
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess(images):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    images = images * 0.5 + 0.5
    
    images = images.clamp(0,1) #gpt recommandation -> needs to be checked
    
    mean = images.new_tensor(IMAGENET_MEAN).view(1,3,1,1)
    std = images.new_tensor(IMAGENET_STD).view(1,3,1,1)
    
    images = (images - mean) / std
    return images.float()

def _extract_images(batch):
    """
    Support datasets returning either a tensor or a tuple/list whose first
    element is the image tensor.
    """
    if torch.is_tensor(batch):
        return batch

    if isinstance(batch, (tuple, list)) and batch:
        if torch.is_tensor(batch[0]):
            return batch[0]

    if isinstance(batch, dict):
        for key in ("image", "images", "x"):
            if key in batch and torch.is_tensor(batch[key]):
                return batch[key]

    raise TypeError(
        "Unsupported DataLoader batch format. Expected an image tensor, "
        "tuple/list beginning with an image tensor, or a dictionary "
        "containing image/images/x."
    )

def _build_inception_model(device):
    """
    Keep pretrained=True for compatibility with older torchvision versions
    commonly used by the original repository.
    """
    model = inception_v3(
        pretrained=True,
    )
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

def calculate_fid(real_loader, generated_loader, device='cuda'):
    # Load pretrained Inception model
    device = torch.device(device)
    model = _build_inception_model(device)

    # Function to get features from a dataloader
    def get_features(loader):
        features = []
        with torch.inference_mode():
            for batch in loader:
                images = _extract_images(batch)
                with torch.no_grad():
                    images = images.to(device,  non_blocking = True)
                    images = preprocess(images)
                    output = model(images)
                    features.append(output.detach().cpu())
                    del images
                    del output
                    
        if not features:
            raise ValueError(
                "Cannot calculate FID from an empty DataLoader."
            )
        return torch.cat(features, dim=0)

    # Extract features
    real_features = get_features(real_loader)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    generated_features = get_features(generated_loader)
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Compute FID score
    fid = FID()
    score = fid.compute_metric(real_features, generated_features)
    return score

if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++
    # Please change the path to your dataset
    real_dataset = ImageDataset(root='dataset/test-ffhq', resolution=256, start_id=0, end_id=100)
    fake_dataset = ImageDataset(root='results/pixel/ffhq/inpainting/samples', resolution=256, start_id=0, end_id=100)

    real_loader = DataLoader(real_dataset, batch_size=8, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=8, shuffle=False)

    fid_score = calculate_fid(real_loader, fake_loader)
    print(f'FID Score: {fid_score.item():.4f}')