"""Seed selection shared by H3 chunk samplers; the queue owns the base seed."""
import hashlib
import copy

SEED_POLICIES = ["fixed_per_generation", "random_per_generation", "fixed_per_chunk", "random_per_chunk"]


def chunk_seed(sampling, index, seed=42, stride=1):
    base = int(sampling.get("seed", seed))
    policy = sampling.get("seed_policy", "fixed_per_generation")
    if policy not in SEED_POLICIES:
        raise ValueError(f"Unknown H3 seed policy: {policy}")
    if policy == "random_per_chunk":
        # A randomized queue seed gives independent, replayable chunk seeds.
        return int.from_bytes(hashlib.sha256(f"h3:{base}:{int(index)}".encode()).digest()[:8], "little")
    if policy == "fixed_per_chunk":
        return (base + int(index) * int(sampling.get("seed_stride", stride))) & 0xFFFFFFFFFFFFFFFF
    return base & 0xFFFFFFFFFFFFFFFF


class WindowNoise:
    """Let the loop's internal windows obey the same visible seed policy."""
    def __init__(self, noise, sampling):
        self.noise = noise
        self.sampling = dict(sampling)
        self.seed = int(noise.seed)

    def generate_noise(self, latent):
        return self.noise.generate_noise(latent)

    def for_window(self, index):
        noise = copy.copy(self.noise)
        noise.seed = chunk_seed(self.sampling, index, self.seed)
        return noise
