import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256): 
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

    def forward(self, x): 
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        return out

class TrackNetTracker:
    def __init__(self, model_path="model_best.pt", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = BallTrackerNet(out_channels=256).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except (RuntimeError, FileNotFoundError):
            try:
                self.model = BallTrackerNet(out_channels=2).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            except FileNotFoundError:
                print(f"Warning: Model weights '{model_path}' not found. Please download them.")
                
        self.model.eval()
        self.frame_buffer = deque(maxlen=3)
        self.width = 640
        self.height = 360

    def postprocess(self, feature_map):
        # Convert index from argmax back to [0, 255] heatmap
        if feature_map.max() <= 1:
            feature_map = feature_map * 255
            
        feature_map = feature_map.reshape((self.height, self.width)).astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        
        # Use HoughCircles as implemented in the original repository to find the ball center
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
        if circles is not None and len(circles) > 0:
            return circles[0][0][0], circles[0][0][1]
        return None

    def track_ball(self, frame):
        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(frame, (self.width, self.height))
        self.frame_buffer.append(img)
        
        # TrackNet requires 3 consecutive frames to make a prediction
        if len(self.frame_buffer) < 3:
            return None

        # Concat: current (t), prev (t-1), preprev (t-2)
        img_current = self.frame_buffer[-1]
        img_prev = self.frame_buffer[-2]
        img_preprev = self.frame_buffer[-3]
        
        imgs = np.concatenate((img_current, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0) # Shape: (9, 360, 640)
        inp = np.expand_dims(imgs, axis=0) # Shape: (1, 9, 360, 640)

        with torch.no_grad():
            inp_tensor = torch.from_numpy(inp).float().to(self.device)
            out = self.model(inp_tensor)
            output = out.argmax(dim=1).detach().cpu().numpy()[0]
            
        coords = self.postprocess(output)
        
        if coords:
            # Scale coordinates back to original video resolution
            scale_x = orig_w / self.width
            scale_y = orig_h / self.height
            return (coords[0] * scale_x, coords[1] * scale_y)
        return None
