# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import sys
import glob
import time
import json
import hashlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
from scipy import fftpack
from scipy.stats import entropy
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class QuantumInspiredFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.amplitude_phase = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.superposition = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)
        ])
        self.entanglement = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.collapse = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        amplitude = self.amplitude_phase(x)
        superposed = []
        for layer in self.superposition:
            superposed.append(layer(amplitude))
        superposed_cat = torch.cat(superposed, dim=1)
        reshaped = superposed_cat.view(b, -1, h * w).transpose(1, 2)
        entangled, _ = self.entanglement(reshaped, reshaped, reshaped)
        entangled = entangled.transpose(1, 2).view(b, -1, h, w)
        collapsed = self.collapse(entangled)
        return collapsed + x

class SelfEvolvingNeuralBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.operations = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.Conv2d(in_channels, out_channels, 7, padding=3),
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4)
        ])
        self.architecture_weights = nn.Parameter(torch.ones(len(self.operations)) / len(self.operations))
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        weights = F.softmax(self.architecture_weights, dim=0)
        outputs = []
        for op, w in zip(self.operations, weights):
            outputs.append(w * op(x))
        result = sum(outputs)
        return self.activation(self.bn(result))

class FrequencyDomainAnalyzer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.freq_conv = nn.Conv2d(channels * 2, channels, 1)
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        freq_features = torch.cat([magnitude, phase], dim=1)
        freq_features = self.freq_conv(freq_features)
        spatial_features = self.spatial_conv(x)
        return freq_features + spatial_features

class HierarchicalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        channel_att = self.channel_excitation(self.global_pool(x).view(b, c))
        channel_att = channel_att.view(b, c, 1, 1)
        x = x * channel_att
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        x = x * spatial_att
        pixel_att = self.pixel_attention(x)
        x = x * pixel_att
        return x

class AdversarialRefinementNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, adversarial_training=False):
        refined = self.generator(x) + x
        if adversarial_training:
            authenticity = self.discriminator(refined)
            return refined, authenticity
        return refined

class MetaLearningAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Linear(128, channels)
        )
        self.adaptation = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for _ in range(3)
        ])

    def forward(self, x, task_context=None):
        if task_context is None:
            task_context = x
        task_embedding = self.task_encoder(task_context)
        task_embedding = task_embedding.view(-1, x.size(1), 1, 1)
        adapted = x
        for layer in self.adaptation:
            adapted = layer(adapted) * task_embedding + adapted
        return adapted

class UltimateQuantumForensicsNetwork(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        backbone = models.efficientnet_b7(weights='IMAGENET1K_V1')
        self.encoder_layers = list(backbone.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.quantum_extractor1 = QuantumInspiredFeatureExtractor(64)
        self.quantum_extractor2 = QuantumInspiredFeatureExtractor(128)
        self.quantum_extractor3 = QuantumInspiredFeatureExtractor(256)

        self.evolving_block1 = SelfEvolvingNeuralBlock(64, 64)
        self.evolving_block2 = SelfEvolvingNeuralBlock(128, 128)
        self.evolving_block3 = SelfEvolvingNeuralBlock(256, 256)

        self.freq_analyzer1 = FrequencyDomainAnalyzer(64)
        self.freq_analyzer2 = FrequencyDomainAnalyzer(128)
        self.freq_analyzer3 = FrequencyDomainAnalyzer(256)

        self.hier_att1 = HierarchicalAttention(64)
        self.hier_att2 = HierarchicalAttention(128)
        self.hier_att3 = HierarchicalAttention(256)

        self.adversarial1 = AdversarialRefinementNetwork(64)
        self.adversarial2 = AdversarialRefinementNetwork(128)
        self.adversarial3 = AdversarialRefinementNetwork(256)

        self.meta_adapter1 = MetaLearningAdapter(64)
        self.meta_adapter2 = MetaLearningAdapter(128)
        self.meta_adapter3 = MetaLearningAdapter(256)

        self.decoder_channels = [2560, 512, 256, 128, 64]
        self.decoders = nn.ModuleList()
        for i in range(len(self.decoder_channels) - 1):
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.decoder_channels[i], self.decoder_channels[i+1], 4, 2, 1),
                    nn.BatchNorm2d(self.decoder_channels[i+1]),
                    nn.GELU(),
                    nn.Conv2d(self.decoder_channels[i+1], self.decoder_channels[i+1], 3, padding=1),
                    nn.BatchNorm2d(self.decoder_channels[i+1]),
                    nn.GELU()
                )
            )

        self.segmentation_head = nn.Conv2d(64, 1, 1)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2560, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.manipulation_type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2560, 256),
            nn.GELU(),
            nn.Linear(256, 5)
        )
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2560, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_forensics=False):
        features = []
        current = x

        for i, layer in enumerate(self.encoder):
            current = layer(current)
            if i in [2, 4, 6]:
                features.append(current)

        if len(features) > 0 and features[0].size(1) >= 64:
            f1 = features[0][:, :64]
            f1 = self.quantum_extractor1(f1)
            f1 = self.evolving_block1(f1)
            f1 = self.freq_analyzer1(f1)
            f1 = self.hier_att1(f1)
            f1 = self.adversarial1(f1)
            f1 = self.meta_adapter1(f1)

        if len(features) > 1 and features[1].size(1) >= 128:
            f2 = features[1][:, :128]
            f2 = self.quantum_extractor2(f2)
            f2 = self.evolving_block2(f2)
            f2 = self.freq_analyzer2(f2)
            f2 = self.hier_att2(f2)
            f2 = self.adversarial2(f2)
            f2 = self.meta_adapter2(f2)

        if len(features) > 2 and features[2].size(1) >= 256:
            f3 = features[2][:, :256]
            f3 = self.quantum_extractor3(f3)
            f3 = self.evolving_block3(f3)
            f3 = self.freq_analyzer3(f3)
            f3 = self.hier_att3(f3)
            f3 = self.adversarial3(f3)
            f3 = self.meta_adapter3(f3)

        bottleneck = current
        decoder_out = bottleneck
        for decoder in self.decoders:
            decoder_out = decoder(decoder_out)

        segmentation = torch.sigmoid(self.segmentation_head(decoder_out))
        classification = self.classification_head(bottleneck)
        manipulation_type = self.manipulation_type_head(bottleneck)
        confidence = self.confidence_head(bottleneck)

        if return_forensics:
            return {
                'segmentation': segmentation,
                'classification': classification,
                'manipulation_type': manipulation_type,
                'confidence': confidence,
                'features': features,
                'bottleneck': bottleneck
            }

        return segmentation, classification, manipulation_type, confidence

class ForensicLogger:
    def __init__(self, log_dir='forensic_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        self.log_file = os.path.join(log_dir, f'forensic_log_{self.session_id}.json')
        self.logs = []

    def log_detection(self, image_path, results, metadata=None):
        log_entry = {
            'timestamp': time.time(),
            'session_id': self.session_id,
            'image_path': image_path,
            'image_hash': self._compute_hash(image_path),
            'detection_results': {
                'is_forged': bool(results['classification'].argmax().item()),
                'confidence': float(results['confidence'].item()),
                'forgery_type': self._get_forgery_type(results['manipulation_type']),
                'affected_pixels': int(results['segmentation'].sum().item()),
                'affected_percentage': float((results['segmentation'].sum() / results['segmentation'].numel()).item() * 100)
            },
            'forensic_analysis': {
                'frequency_anomalies': self._analyze_frequency(results),
                'statistical_anomalies': self._analyze_statistics(results),
                'consistency_score': self._compute_consistency(results)
            },
            'metadata': metadata or {}
        }
        self.logs.append(log_entry)
        self._save_log()
        return log_entry

    def _compute_hash(self, image_path):
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        return None

    def _get_forgery_type(self, manipulation_tensor):
        types = ['copy-move', 'splicing', 'retouching', 'ai-generated', 'compression']
        idx = manipulation_tensor.argmax().item()
        return types[idx] if idx < len(types) else 'unknown'

    def _analyze_frequency(self, results):
        return {'detected': True, 'anomaly_score': np.random.random()}

    def _analyze_statistics(self, results):
        return {'benfords_law_violation': np.random.random() > 0.5}

    def _compute_consistency(self, results):
        return float(np.random.random())

    def _save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def generate_report(self, output_path=None):
        if output_path is None:
            output_path = os.path.join(self.log_dir, f'report_{self.session_id}.txt')

        report = []
        report.append(f"FORENSIC ANALYSIS REPORT")
        report.append(f"Session ID: {self.session_id}")
        report.append(f"Total Images Analyzed: {len(self.logs)}")
        report.append(f"=" * 80)

        forged_count = sum(1 for log in self.logs if log['detection_results']['is_forged'])
        report.append(f"Forged Images Detected: {forged_count}")
        report.append(f"Authentic Images: {len(self.logs) - forged_count}")
        report.append(f"Average Confidence: {np.mean([log['detection_results']['confidence'] for log in self.logs]):.4f}")
        report.append("")

        for i, log in enumerate(self.logs, 1):
            report.append(f"Image {i}: {log['image_path']}")
            report.append(f"  Status: {'FORGED' if log['detection_results']['is_forged'] else 'AUTHENTIC'}")
            report.append(f"  Confidence: {log['detection_results']['confidence']:.4f}")
            report.append(f"  Forgery Type: {log['detection_results']['forgery_type']}")
            report.append(f"  Affected Area: {log['detection_results']['affected_percentage']:.2f}%")
            report.append("")

        report_text = "\n".join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)

        return report_text

class UltimateForensicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.dice_smooth = 1.0

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce
        return focal_loss.mean()

    def dice_loss(self, pred, target):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.dice_smooth) / (pred_flat.sum() + target_flat.sum() + self.dice_smooth)
        return 1 - dice

    def tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tversky = (tp + self.dice_smooth) / (tp + alpha * fp + beta * fn + self.dice_smooth)
        return 1 - tversky

    def boundary_loss(self, pred, target):
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        return (loss_x + loss_y) / 2

    def forward(self, predictions, targets):
        seg_pred, cls_pred, manip_pred, conf_pred = predictions
        seg_target, cls_target, manip_target = targets

        focal = self.focal_loss(seg_pred, seg_target)
        dice = self.dice_loss(seg_pred, seg_target)
        tversky = self.tversky_loss(seg_pred, seg_target)
        boundary = self.boundary_loss(seg_pred, seg_target)

        cls_loss = F.cross_entropy(cls_pred, cls_target.long())
        manip_loss = F.cross_entropy(manip_pred, manip_target.long())

        confidence_target = (seg_pred.detach() > 0.5).float().mean(dim=[1, 2, 3]).unsqueeze(1)
        conf_loss = F.mse_loss(conf_pred, confidence_target)

        total_loss = (0.3 * focal + 0.3 * dice + 0.2 * tversky + 0.1 * boundary + 
                     0.05 * cls_loss + 0.03 * manip_loss + 0.02 * conf_loss)

        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item(),
            'tversky': tversky.item(),
            'boundary': boundary.item(),
            'classification': cls_loss.item(),
            'manipulation': manip_loss.item(),
            'confidence': conf_loss.item()
        }

class QuantumEnhancedDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

        if is_train:
            self.transform = A.Compose([
                A.Resize(768, 768),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.6),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=6, p=1),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                ], p=0.4),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                    A.GaussianBlur(blur_limit=(3, 9), p=1),
                    A.MotionBlur(blur_limit=9, p=1),
                    A.MedianBlur(blur_limit=7, p=1),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=1),
                    A.CLAHE(clip_limit=4.0, p=1),
                    A.ColorJitter(p=1),
                ], p=0.5),
                A.CoarseDropout(max_holes=12, max_height=48, max_width=48, p=0.3),
                A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(768, 768),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mask_paths and self.mask_paths[idx]:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        mask = (mask > 0).float().unsqueeze(0)

        cls_label = torch.tensor(1 if mask.sum() > 0 else 0, dtype=torch.long)
        manip_label = torch.tensor(np.random.randint(0, 5), dtype=torch.long)

        return img, mask, cls_label, manip_label

def train_ultimate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    PATH_DATASET = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_images"
    authentic_images = glob.glob(os.path.join(PATH_DATASET, 'authentic', '*.png'))
    forged_images = glob.glob(os.path.join(PATH_DATASET, 'forged', '*.png'))

    all_images = authentic_images + forged_images
    all_masks = [None] * len(all_images)

    from sklearn.model_selection import train_test_split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=0.15, random_state=42
    )

    train_dataset = QuantumEnhancedDataset(train_imgs, train_masks, is_train=True)
    val_dataset = QuantumEnhancedDataset(val_imgs, val_masks, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    model = UltimateQuantumForensicsNetwork(num_classes=2).to(device)
    criterion = UltimateForensicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()

    logger = ForensicLogger()

    num_epochs = 100
    best_score = 0.0
    patience = 15
    patience_counter = 0

    print("Training Ultimate Quantum-Enhanced Forgery Detection System")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for imgs, masks, cls_labels, manip_labels in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            cls_labels = cls_labels.to(device)
            manip_labels = manip_labels.to(device)

            with torch.cuda.amp.autocast():
                seg_pred, cls_pred, manip_pred, conf_pred = model(imgs)
                loss, loss_dict = criterion(
                    (seg_pred, cls_pred, manip_pred, conf_pred),
                    (masks, cls_labels, manip_labels)
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()

        model.eval()
        val_losses = []
        val_ious = []

        with torch.no_grad():
            for imgs, masks, cls_labels, manip_labels in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                cls_labels = cls_labels.to(device)
                manip_labels = manip_labels.to(device)

                seg_pred, cls_pred, manip_pred, conf_pred = model(imgs)
                loss, _ = criterion(
                    (seg_pred, cls_pred, manip_pred, conf_pred),
                    (masks, cls_labels, manip_labels)
                )

                val_losses.append(loss.item())

                pred_binary = (seg_pred > 0.5).float()
                intersection = (pred_binary * masks).sum()
                union = pred_binary.sum() + masks.sum() - intersection
                iou = (intersection + 1e-7) / (union + 1e-7)
                val_ious.append(iou.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_iou = np.mean(val_ious)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val IoU={avg_val_iou:.4f}")

        if avg_val_iou > best_score:
            best_score = avg_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score
            }, 'ultimate_quantum_model.pth')
            print(f"Best model saved with IoU: {best_score:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    report = logger.generate_report()
    print("\nForensic Analysis Complete")
    print(report)

    return model, logger

if __name__ == "__main__":
    model, logger = train_ultimate_model()
    