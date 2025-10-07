"""
Tests GPU AMD - Vérification ROCm et performances
Tests pour AMD Radeon RX 7800 XT avec ROCm

Author: Claude Code
Date: 2025-10-06
"""

import pytest
import time
import numpy as np

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@pytest.mark.gpu
class TestAMDGPU:
    """Tests pour GPU AMD avec ROCm"""

    def test_torch_installed(self):
        """Vérifie que PyTorch est installé"""
        assert TORCH_AVAILABLE, "PyTorch not installed - run setup_pytorch_rocm.sh"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_gpu_available(self):
        """Vérifie que GPU AMD est disponible via ROCm"""
        assert torch.cuda.is_available(), (
            "GPU not available. Check:\n"
            "1. ROCm installed (rocm-smi)\n"
            "2. HSA_OVERRIDE_GFX_VERSION=11.0.0 set\n"
            "3. PyTorch ROCm version installed"
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    def test_gpu_name(self):
        """Vérifie nom du GPU"""
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n   GPU: {gpu_name}")

        # AMD GPU devrait contenir "AMD" ou "Radeon"
        assert ("AMD" in gpu_name or "Radeon" in gpu_name or "gfx" in gpu_name), \
            f"Unexpected GPU: {gpu_name}"

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    def test_gpu_memory(self):
        """Vérifie VRAM disponible (>10 GB pour 7800 XT)"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n   VRAM: {vram_total:.1f} GB")

        assert vram_total > 10, \
            f"VRAM insuffisante: {vram_total:.1f} GB (attendu > 10 GB pour RX 7800 XT)"

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    def test_fp16_support(self):
        """Teste support FP16 (half precision)"""
        device = torch.device("cuda:0")

        # Créer modèle simple
        model = torch.nn.Linear(10, 10).to(device)
        model = model.half()  # Convert to FP16

        # Test inference
        x = torch.randn(1, 10).half().to(device)
        with torch.no_grad():
            y = model(x)

        assert y.dtype == torch.float16, "FP16 non supporté correctement"
        print("\n   ✅ FP16 supporté")

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    def test_tensor_operations(self):
        """Teste opérations tenseur GPU vs CPU"""
        size = 1000
        device = torch.device("cuda:0")

        # CPU
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)

        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start

        # GPU
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)

        # Warmup
        for _ in range(3):
            _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()

        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time
        print(f"\n   CPU: {cpu_time*1000:.1f}ms, GPU: {gpu_time*1000:.1f}ms, Speedup: {speedup:.1f}x")

        assert gpu_time < cpu_time, "GPU devrait être plus rapide que CPU"
        assert speedup > 2, f"Speedup trop faible: {speedup:.1f}x (attendu >2x)"

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    @pytest.mark.slow
    def test_yolo_inference_speed(self):
        """Benchmark inference YOLO sur GPU"""
        try:
            from ultralytics import YOLO
        except ImportError:
            pytest.skip("ultralytics not installed")

        device = torch.device("cuda:0")

        # Load YOLOv8n (nano - fastest)
        model = YOLO("yolov8n.pt")
        model.to(device)

        # Créer image test
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(5):
            _ = model(img, device=device, verbose=False)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            _ = model(img, device=device, verbose=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time

        print(f"\n   YOLOv8n:")
        print(f"   Inference: {avg_time*1000:.2f}ms")
        print(f"   FPS: {fps:.1f}")

        # Target: >80 FPS pour YOLOv8n sur RX 7800 XT
        assert fps > 30, f"FPS trop bas: {fps:.1f} (attendu >30)"

        if fps < 60:
            print("   ⚠️  FPS < 60, vérifier optimisations ROCm")

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    def test_vram_management(self):
        """Teste gestion VRAM"""
        device = torch.device("cuda:0")

        def get_vram_usage():
            """Retourne VRAM utilisée en GB"""
            return torch.cuda.memory_allocated() / 1e9

        # Initial
        torch.cuda.empty_cache()
        initial_vram = get_vram_usage()

        # Allouer tenseur
        large_tensor = torch.randn(10000, 10000, device=device)
        allocated_vram = get_vram_usage()

        print(f"\n   Initial VRAM: {initial_vram:.2f} GB")
        print(f"   After allocation: {allocated_vram:.2f} GB")

        # Vérifier allocation
        assert allocated_vram > initial_vram, "VRAM devrait augmenter après allocation"

        # Libérer
        del large_tensor
        torch.cuda.empty_cache()
        final_vram = get_vram_usage()

        print(f"   After cleanup: {final_vram:.2f} GB")

        # VRAM devrait être proche de initial
        assert final_vram < allocated_vram, "VRAM devrait diminuer après cleanup"

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    def test_multi_batch_inference(self):
        """Teste inference avec différents batch sizes"""
        device = torch.device("cuda:0")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 10)
        ).to(device).eval()

        results = {}

        for batch_size in [1, 2, 4, 8]:
            # Create input
            x = torch.randn(batch_size, 3, 224, 224, device=device)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x)
            torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(20):
                start = time.time()
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            throughput = batch_size / avg_time

            results[batch_size] = {
                "time": avg_time * 1000,  # ms
                "throughput": throughput
            }

        print("\n   Batch Size Performance:")
        for bs, result in results.items():
            print(f"   BS={bs}: {result['time']:.2f}ms, {result['throughput']:.1f} img/s")

        # Batch size 1 devrait être le plus rapide en latency
        assert results[1]["time"] < results[8]["time"], \
            "Batch size 1 devrait avoir latency plus faible"


@pytest.mark.gpu
class TestGPUOptimization:
    """Tests d'optimisation GPU"""

    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                       reason="GPU not available")
    def test_fp16_vs_fp32_performance(self):
        """Compare performance FP16 vs FP32"""
        device = torch.device("cuda:0")

        model = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        ).to(device).eval()

        x = torch.randn(32, 512, device=device)

        # FP32
        times_fp32 = []
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            times_fp32.append(time.time() - start)

        avg_fp32 = sum(times_fp32) / len(times_fp32)

        # FP16
        model_fp16 = model.half()
        x_fp16 = x.half()

        times_fp16 = []
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = model_fp16(x_fp16)
            torch.cuda.synchronize()
            times_fp16.append(time.time() - start)

        avg_fp16 = sum(times_fp16) / len(times_fp16)

        speedup = avg_fp32 / avg_fp16

        print(f"\n   FP32: {avg_fp32*1000:.2f}ms")
        print(f"   FP16: {avg_fp16*1000:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x")

        # FP16 devrait être plus rapide
        assert avg_fp16 < avg_fp32, "FP16 devrait être plus rapide que FP32"
        assert speedup > 1.2, f"Speedup trop faible: {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
