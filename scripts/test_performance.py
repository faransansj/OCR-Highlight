"""
Automated Test Suite for OCR-Highlight 2.0
Tests all new modules and generates performance reports
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from symbols.line_detector import LineMarkupDetector
from symbols.shape_detector import ShapeDetector
from data_generation.synthetic_generator import SyntheticDataGenerator


class PerformanceMetrics:
    """Calculate detection performance metrics"""
    
    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_metrics(predictions: List[Dict], 
                         ground_truths: List[Dict],
                         iou_threshold: float = 0.5) -> Dict:
        """
        Calculate precision, recall, F1, mIoU
        
        Args:
            predictions: List of predicted bboxes
            ground_truths: List of GT bboxes
            iou_threshold: Minimum IoU for true positive
        
        Returns:
            Dict with metrics
        """
        if len(ground_truths) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'miou': 0.0,
                'tp': 0,
                'fp': len(predictions),
                'fn': 0
            }
        
        if len(predictions) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'miou': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(ground_truths)
            }
        
        # Match predictions to GTs
        matched_gt = set()
        true_positives = 0
        total_iou = 0.0
        
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                iou = PerformanceMetrics.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                true_positives += 1
                total_iou += best_iou
                matched_gt.add(best_gt_idx)
        
        false_positives = len(predictions) - true_positives
        false_negatives = len(ground_truths) - true_positives
        
        precision = true_positives / len(predictions) if len(predictions) > 0 else 0.0
        recall = true_positives / len(ground_truths) if len(ground_truths) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        miou = total_iou / true_positives if true_positives > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'miou': miou,
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives
        }


class TestRunner:
    """Run all tests and generate reports"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def test_line_detector(self, num_samples: int = 200) -> Dict:
        """Test line detector on synthetic data"""
        print(f"\n{'='*60}")
        print("TEST 1: Line Detector (Underlines & Strikethroughs)")
        print(f"{'='*60}")
        
        # Generate test data
        print(f"Generating {num_samples} synthetic line samples...")
        generator = SyntheticDataGenerator(output_dir=str(self.output_dir / "line_test_data"))
        samples = generator.generate_batch(
            count=num_samples,
            markup_types=['underline', 'strikethrough'],
            max_markups_per_image=2
        )
        
        # Test detector
        print("Running line detector...")
        detector = LineMarkupDetector()
        
        all_predictions = []
        all_ground_truths = []
        processing_times = []
        
        for sample in samples:
            # Load image
            img = cv2.imread(sample.image_path)
            
            # Detect
            start_time = time.time()
            detections = detector.detect(img)
            processing_times.append(time.time() - start_time)
            
            # Convert to comparable format
            predictions = [
                {'bbox': d.bbox, 'type': d.type, 'subtype': d.subtype}
                for d in detections
            ]
            
            ground_truths = [
                {'bbox': a.bbox, 'type': a.markup_type, 'subtype': a.subtype}
                for a in sample.annotations
                if a.markup_type in ['underline', 'strikethrough']
            ]
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_metrics(all_predictions, all_ground_truths)
        metrics['avg_processing_time'] = np.mean(processing_times)
        metrics['total_samples'] = num_samples
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Precision: {metrics['precision']:.4f} (target: > 0.80)")
        print(f"  Recall:    {metrics['recall']:.4f} (target: > 0.70)")
        print(f"  F1-Score:  {metrics['f1']:.4f} (target: > 0.70)")
        print(f"  mIoU:      {metrics['miou']:.4f} (target: > 0.75)")
        print(f"  Avg Time:  {metrics['avg_processing_time']*1000:.2f}ms/image")
        print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        
        # Pass/Fail
        passed = (metrics['precision'] >= 0.70 and 
                 metrics['recall'] >= 0.60 and 
                 metrics['miou'] >= 0.65)
        
        print(f"\n{'✅ PASS' if passed else '❌ FAIL'} (MVP criteria)")
        
        self.results['line_detector'] = metrics
        return metrics
    
    def test_shape_detector(self, num_samples: int = 200) -> Dict:
        """Test shape detector on synthetic data"""
        print(f"\n{'='*60}")
        print("TEST 2: Shape Detector (Circles, Rectangles, etc.)")
        print(f"{'='*60}")
        
        # Generate test data
        print(f"Generating {num_samples} synthetic shape samples...")
        generator = SyntheticDataGenerator(output_dir=str(self.output_dir / "shape_test_data"))
        samples = generator.generate_batch(
            count=num_samples,
            markup_types=['circle', 'rectangle'],
            max_markups_per_image=2
        )
        
        # Test detector
        print("Running shape detector...")
        detector = ShapeDetector()
        
        all_predictions = []
        all_ground_truths = []
        processing_times = []
        
        per_shape_stats = {'circle': {'tp': 0, 'fp': 0, 'fn': 0},
                          'rectangle': {'tp': 0, 'fp': 0, 'fn': 0}}
        
        for sample in samples:
            # Load image
            img = cv2.imread(sample.image_path)
            
            # Detect
            start_time = time.time()
            detections = detector.detect(img)
            processing_times.append(time.time() - start_time)
            
            # Convert to comparable format
            predictions = [
                {'bbox': d.bbox, 'type': d.shape_type}
                for d in detections
            ]
            
            ground_truths = [
                {'bbox': a.bbox, 'type': a.markup_type}
                for a in sample.annotations
                if a.markup_type in ['circle', 'rectangle']
            ]
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_metrics(all_predictions, all_ground_truths)
        metrics['avg_processing_time'] = np.mean(processing_times)
        metrics['total_samples'] = num_samples
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Precision: {metrics['precision']:.4f} (target: > 0.75)")
        print(f"  Recall:    {metrics['recall']:.4f} (target: > 0.65)")
        print(f"  F1-Score:  {metrics['f1']:.4f} (target: > 0.65)")
        print(f"  mIoU:      {metrics['miou']:.4f} (target: > 0.70)")
        print(f"  Avg Time:  {metrics['avg_processing_time']*1000:.2f}ms/image")
        print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        
        # Pass/Fail
        passed = (metrics['precision'] >= 0.65 and 
                 metrics['recall'] >= 0.55 and 
                 metrics['miou'] >= 0.60)
        
        print(f"\n{'✅ PASS' if passed else '❌ FAIL'} (MVP criteria)")
        
        self.results['shape_detector'] = metrics
        return metrics
    
    def test_data_generator(self, num_samples: int = 1000) -> Dict:
        """Test synthetic data generator"""
        print(f"\n{'='*60}")
        print("TEST 3: Synthetic Data Generator")
        print(f"{'='*60}")
        
        print(f"Generating {num_samples} samples...")
        generator = SyntheticDataGenerator(output_dir=str(self.output_dir / "generator_test"))
        
        start_time = time.time()
        samples = generator.generate_batch(
            count=num_samples,
            markup_types=['highlight', 'underline', 'strikethrough', 'circle', 'rectangle'],
            max_markups_per_image=3
        )
        total_time = time.time() - start_time
        
        # Analyze results
        languages = set(s.metadata['language'] for s in samples)
        markup_types = set()
        total_annotations = 0
        
        for s in samples:
            total_annotations += len(s.annotations)
            for a in s.annotations:
                markup_types.add(a.markup_type)
        
        metrics = {
            'total_samples': len(samples),
            'total_time': total_time,
            'samples_per_second': len(samples) / total_time,
            'languages': len(languages),
            'markup_types': len(markup_types),
            'avg_annotations_per_image': total_annotations / len(samples),
            'gt_accuracy': 1.0  # Perfect by design
        }
        
        print(f"\n📊 Results:")
        print(f"  Samples Generated: {metrics['total_samples']}")
        print(f"  Total Time:        {metrics['total_time']:.2f}s")
        print(f"  Speed:             {metrics['samples_per_second']:.2f} images/sec")
        print(f"  Languages:         {metrics['languages']} (target: 4+)")
        print(f"  Markup Types:      {metrics['markup_types']} (target: 5+)")
        print(f"  Avg Annotations:   {metrics['avg_annotations_per_image']:.2f}")
        print(f"  GT Accuracy:       100% (auto-generated)")
        
        # Pass/Fail
        passed = (metrics['samples_per_second'] >= 1.5 and  # >100/min
                 metrics['languages'] >= 4 and
                 metrics['markup_types'] >= 5)
        
        print(f"\n{'✅ PASS' if passed else '❌ FAIL'} (target: >100 images/min)")
        
        self.results['data_generator'] = metrics
        return metrics
    
    def generate_report(self):
        """Generate final test report"""
        print(f"\n{'='*60}")
        print("FINAL TEST REPORT")
        print(f"{'='*60}\n")
        
        # Save to JSON
        report_path = self.output_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_path}")
        
        # Summary
        print("\n🎯 Performance Summary:")
        
        if 'line_detector' in self.results:
            r = self.results['line_detector']
            status = "✅" if r['miou'] >= 0.65 else "❌"
            print(f"  {status} Line Detector:  mIoU={r['miou']:.4f}, F1={r['f1']:.4f}")
        
        if 'shape_detector' in self.results:
            r = self.results['shape_detector']
            status = "✅" if r['miou'] >= 0.60 else "❌"
            print(f"  {status} Shape Detector: mIoU={r['miou']:.4f}, F1={r['f1']:.4f}")
        
        if 'data_generator' in self.results:
            r = self.results['data_generator']
            status = "✅" if r['samples_per_second'] >= 1.5 else "❌"
            print(f"  {status} Data Generator: {r['samples_per_second']:.2f} imgs/sec")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Run all tests
    runner = TestRunner(output_dir="test_results")
    
    # Test 1: Line Detector
    runner.test_line_detector(num_samples=200)
    
    # Test 2: Shape Detector
    runner.test_shape_detector(num_samples=200)
    
    # Test 3: Data Generator
    runner.test_data_generator(num_samples=1000)
    
    # Final report
    runner.generate_report()
