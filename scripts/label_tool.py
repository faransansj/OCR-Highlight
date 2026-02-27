#!/usr/bin/env python3
"""
Simple Artifact Labeling Tool (Phase 2.2)
Help Sensei label real document images for Milestone 2
"""

import cv2
import json
import os
import sys

class Labeler:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.image = cv2.imread(image_path)
        self.temp_image = self.image.copy()
        self.annotations = []
        self.current_bbox = None
        self.drawing = False
        
        if self.image is None:
            print(f"Error: Could not load image {image_path}")
            sys.exit(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_bbox = [x, y, 0, 0]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_image = self.image.copy()
                # Draw previous annotations
                self._draw_all()
                cv2.rectangle(self.temp_image, (self.current_bbox[0], self.current_bbox[1]), (x, y), (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            w = x - self.current_bbox[0]
            h = y - self.current_bbox[1]
            if w != 0 and h != 0:
                # Store as [x, y, w, h]
                # Normalize if drawn backwards
                nx = min(x, self.current_bbox[0])
                ny = min(y, self.current_bbox[1])
                nw = abs(w)
                nh = abs(h)
                
                print(f"Region captured: [{nx}, {ny}, {nw}, {nh}]")
                color = input("Enter color (y/g/p) or cancel (c): ").lower()
                
                if color in ['y', 'g', 'p']:
                    color_map = {'y': 'yellow', 'g': 'green', 'p': 'pink'}
                    self.annotations.append({
                        'color': color_map[color],
                        'bbox': [nx, ny, nw, nh]
                    })
                    self._draw_all()
                else:
                    print("Annotation cancelled.")
                    self.temp_image = self.image.copy()
                    self._draw_all()

    def _draw_all(self):
        for ann in self.annotations:
            color = (0, 255, 255) if ann['color'] == 'yellow' else (0, 255, 0)
            if ann['color'] == 'pink': color = (255, 0, 255)
            x, y, w, h = ann['bbox']
            cv2.rectangle(self.temp_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(self.temp_image, ann['color'], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def run(self):
        cv2.namedWindow("Aris Labeling Tool")
        cv2.setMouseCallback("Aris Labeling Tool", self.mouse_callback)
        
        print("\n=== Aris Labeling Tool ===")
        print("1. Drag mouse to draw a box around a highlight.")
        print("2. Enter 'y' (yellow), 'g' (green), or 'p' (pink) in terminal.")
        print("3. Press 's' to save and exit, 'q' to quit without saving.")
        
        while True:
            cv2.imshow("Aris Labeling Tool", self.temp_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.save()
                break
            elif key == ord('q'):
                print("Exiting without saving.")
                break
        
        cv2.destroyAllWindows()

    def save(self):
        data = {
            'image_path': self.image_path,
            'highlight_annotations': self.annotations
        }
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {self.output_path}")

if __name__ == "__main__":
    # This tool is intended to be run locally by Sensei
    print("This script provides a GUI for manual labeling.")
    print("Usage: python3 label_tool.py <image_path> <output_json_path>")
