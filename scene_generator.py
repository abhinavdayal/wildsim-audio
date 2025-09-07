#!/usr/bin/env python3
"""
Automated Scene Configuration Generator

Generates positive/negative elephant detection scenes with:
- Random directional sounds (10-50) excluding elephants
- Random ambient sounds (3-5) 
- Physics-based positioning with height constraints
- Configurable positive/negative sample ratio
- JSONL output format for batch processing
- Optional audio rendering from JSONL configurations
"""

import json
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
import sys
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_manager import DatasetManager
from simple_simulator import SimpleSimulator, load_config_from_yaml
from simple_config import SceneConfig, SoundPlacement, AmbientSound

@dataclass
class GeneratorConfig:
    """Configuration for the scene generator"""
    num_samples: int
    positive_ratio: float  # 0.0 to 1.0 - percentage of positive samples
    min_directional_sounds: int = 10
    max_directional_sounds: int = 50
    min_ambient_sounds: int = 3
    max_ambient_sounds: int = 5
    min_distance: float = 10.0
    max_distance: float = 500.0
    scene_duration: float = 30.0
    sample_rate: int = 16000
    output_dir: str = "outputs/generated_scenes"

class SceneGenerator:
    """Generates randomized elephant detection scenes"""
    
    # Ambient sound types that should be treated as background
    AMBIENT_TYPES = {
        "rain", "thunderstorm", "waterdrops", "wind", "fire", 
        "silence", "insect", "water", "birdchirping"
    }
    
    # Directional sound types (excluding elephant)
    DIRECTIONAL_TYPES = {
        "axe", "bird", "chainsaw", "clapping", "firework", "footsteps", 
        "frog", "generator", "gunshot", "handsaw", "helicopter", "lion",
        "speaking", "squirrel", "treefalling", "vehicleengine", "whistling",
        "wingflaping", "wolfhowl", "woodchop", "bear", "cat", "chicken",
        "cow", "dog", "dolphin", "donkey", "horse", "sheep", "monkey"
    }
    
    # Height constraints for different sound types (in meters)
    HEIGHT_CONSTRAINTS = {
        # Aerial sounds
        "helicopter": (20.0, 100.0),
        "bird": (5.0, 50.0),
        "wingflaping": (3.0, 30.0),
        
        # Tree-level sounds  
        "monkey": (3.0, 20.0),
        "squirrel": (1.0, 15.0),
        "treefalling": (0.0, 5.0),
        
        # Ground-level animals
        "lion": (0.0, 3.0),
        "bear": (0.0, 3.0),
        "elephant": (0.0, 4.0),
        "cow": (0.0, 2.0),
        "horse": (0.0, 2.5),
        "dog": (0.0, 1.5),
        "cat": (0.0, 1.0),
        "sheep": (0.0, 1.0),
        "chicken": (0.0, 1.0),
        "frog": (0.0, 0.5),
        
        # Human activities
        "speaking": (1.0, 2.5),
        "clapping": (1.0, 2.5),
        "footsteps": (0.0, 0.2),
        "whistling": (1.0, 2.5),
        
        # Machinery/tools
        "chainsaw": (0.5, 3.0),
        "axe": (0.5, 2.5),
        "handsaw": (0.5, 2.5),
        "woodchop": (0.5, 2.5),
        "generator": (0.0, 2.0),
        "vehicleengine": (0.0, 3.0),
        
        # Other sounds
        "gunshot": (1.0, 5.0),
        "firework": (0.0, 50.0),
        "dolphin": (0.0, 2.0),  # Assuming water surface level
        "donkey": (0.0, 2.0),
        
        # Default for unknown types
        "default": (0.0, 5.0)
    }
    
    def __init__(self, dataset_manager: DatasetManager, config: GeneratorConfig):
        self.dataset_manager = dataset_manager
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache available files by sound type
        self._build_sound_type_cache()
    
    def _build_sound_type_cache(self):
        """Build cache of available audio files by sound type"""
        self.sound_cache = {}
        
        print("🔍 Building sound type cache...")
        for file_hash, file_info in self.dataset_manager.catalog.catalog["files"].items():
            source_type = file_info.get("source_type", "unknown").lower()
            file_path = file_info.get("absolute_path")
            
            if source_type not in self.sound_cache:
                self.sound_cache[source_type] = []
            
            self.sound_cache[source_type].append(file_path)
        
        print(f"✅ Cache built with {len(self.sound_cache)} sound types:")
        for sound_type, files in self.sound_cache.items():
            print(f"   - {sound_type}: {len(files)} files")
    
    def _get_physics_based_height(self, sound_type: str) -> float:
        """Get realistic height for a sound type based on physics"""
        constraints = self.HEIGHT_CONSTRAINTS.get(sound_type, self.HEIGHT_CONSTRAINTS["default"])
        min_height, max_height = constraints
        return random.uniform(min_height, max_height)
    
    def _generate_random_position(self, sound_type: str, distance: float) -> Tuple[float, float, float]:
        """Generate random 3D position with physics-based height constraints"""
        # Random angle for azimuth
        azimuth_rad = random.uniform(0, 2 * math.pi)
        
        # Calculate x, y from distance and azimuth
        x = distance * math.cos(azimuth_rad)
        y = distance * math.sin(azimuth_rad)
        
        # Get physics-based height
        z = self._get_physics_based_height(sound_type)
        
        return x, y, z
    
    def _select_random_audio_file(self, sound_type: str) -> Optional[str]:
        """Select a random audio file for the given sound type"""
        if sound_type in self.sound_cache and self.sound_cache[sound_type]:
            return random.choice(self.sound_cache[sound_type])
        
        # Try variations of the sound type name
        for cached_type in self.sound_cache.keys():
            if sound_type in cached_type or cached_type in sound_type:
                return random.choice(self.sound_cache[cached_type])
        
        print(f"⚠️  No audio files found for sound type: {sound_type}")
        return None
    
    def _generate_directional_sounds(self, num_sounds: int, exclude_elephant: bool = True) -> List[Dict]:
        """Generate random directional sounds"""
        sounds = []
        available_types = list(self.DIRECTIONAL_TYPES)
        
        if exclude_elephant:
            available_types = [t for t in available_types if "elephant" not in t]
        
        for _ in range(num_sounds):
            sound_type = random.choice(available_types)
            # Use different distance ranges based on sound type for better audibility
            if sound_type in ['elephant', 'lion', 'bear', 'wolf', 'monkey']:
                # Large animals - can be heard from farther away
                distance = random.uniform(20.0, 150.0)
            elif sound_type in ['bird', 'chicken', 'cat', 'dog', 'sheep', 'horse', 'cow']:
                # Smaller animals and domestic animals - closer for audibility
                distance = random.uniform(15.0, 100.0)
            else:
                # Other sounds - moderate distances
                distance = random.uniform(self.config.min_distance, min(self.config.max_distance, 200.0))
            x, y, z = self._generate_random_position(sound_type, distance)
            
            audio_file = self._select_random_audio_file(sound_type)
            if not audio_file:
                continue  # Skip if no audio file available
            
            # Random start time throughout the scene
            start_time = random.uniform(0.0, max(0.0, self.config.scene_duration - 5.0))
            # Use higher base volumes for animal sounds
            if sound_type in ['elephant', 'lion', 'bear', 'wolf']:
                # Large animal calls are naturally loud
                volume = random.uniform(1.2, 2.0)
            elif sound_type in ['bird', 'monkey', 'dog', 'cat', 'chicken', 'horse', 'cow', 'sheep']:
                # Other animals - good audible range
                volume = random.uniform(0.8, 1.6)
            else:
                # Non-animal sounds - standard range
                volume = random.uniform(0.6, 1.3)
            
            sound = {
                "sound_type": sound_type,
                "audio_file": audio_file,
                "position": [x, y, z],
                "distance": distance,
                "azimuth": math.degrees(math.atan2(y, x)) % 360,
                "elevation": math.degrees(math.atan2(z, math.sqrt(x*x + y*y))),
                "start_time": start_time,
                "volume": volume
            }
            sounds.append(sound)
        
        return sounds
    
    def _generate_ambient_sounds(self, num_sounds: int) -> List[Dict]:
        """Generate random ambient background sounds"""
        ambient_sounds = []
        available_types = list(self.AMBIENT_TYPES)
        
        for _ in range(num_sounds):
            ambient_type = random.choice(available_types)
            # Reduce ambient levels to not overwhelm animal sounds
            level_db = random.uniform(-50.0, -35.0)  # Quieter background levels
            
            # Try to find specific audio file, otherwise use procedural
            audio_file = self._select_random_audio_file(ambient_type)
            
            ambient = {
                "sound_type": ambient_type,
                "level_db": level_db,
                "audio_file": audio_file  # May be None for procedural
            }
            ambient_sounds.append(ambient)
        
        return ambient_sounds
    
    def _generate_elephant_group(self, group_size: int = None) -> List[Dict]:
        """Generate a group of elephants with realistic positioning"""
        if group_size is None:
            group_size = random.randint(1, 5)
        
        elephants = []
        
        # Central position for the group
        # Keep elephant groups closer for better audibility
        base_distance = random.uniform(30.0, 120.0)
        base_x, base_y, base_z = self._generate_random_position("elephant", base_distance)
        
        for i in range(group_size):
            # Elephants in a group stay within ~20m of each other
            offset_x = random.uniform(-20.0, 20.0)
            offset_y = random.uniform(-20.0, 20.0)
            
            x = base_x + offset_x
            y = base_y + offset_y
            z = self._get_physics_based_height("elephant")
            distance = math.sqrt(x*x + y*y)
            
            audio_file = self._select_random_audio_file("elephant")
            if not audio_file:
                continue
            
            # Elephants might call at different times
            start_time = random.uniform(0.0, max(0.0, self.config.scene_duration - 8.0))
            # Elephants should be loud and prominent
            volume = random.uniform(1.4, 2.2)
            
            elephant = {
                "sound_type": "elephant",
                "audio_file": audio_file,
                "position": [x, y, z],
                "distance": distance,
                "azimuth": math.degrees(math.atan2(y, x)) % 360,
                "elevation": math.degrees(math.atan2(z, math.sqrt(x*x + y*y))),
                "start_time": start_time,
                "volume": volume
            }
            elephants.append(elephant)
        
        return elephants
    
    def generate_scene(self, is_positive: bool, scene_id: int) -> Dict:
        """Generate a single scene configuration"""
        # Random number of sounds
        num_directional = random.randint(
            self.config.min_directional_sounds,
            self.config.max_directional_sounds
        )
        num_ambient = random.randint(
            self.config.min_ambient_sounds,
            self.config.max_ambient_sounds
        )
        
        # Generate base sounds (no elephants)
        directional_sounds = self._generate_directional_sounds(num_directional)
        ambient_sounds = self._generate_ambient_sounds(num_ambient)
        
        # Add elephants for positive samples
        if is_positive:
            elephant_sounds = self._generate_elephant_group()
            directional_sounds.extend(elephant_sounds)
        
        # Create scene configuration
        scene = {
            "name": f"Generated_Scene_{scene_id:04d}_{'positive' if is_positive else 'negative'}",
            "duration": self.config.scene_duration,
            "sample_rate": self.config.sample_rate,
            "sounds": directional_sounds,
            "ambient": ambient_sounds,
            "metadata": {
                "is_positive": is_positive,
                "scene_id": scene_id,
                "num_directional_sounds": len(directional_sounds),
                "num_ambient_sounds": len(ambient_sounds),
                "contains_elephant": is_positive,
                "elephant_count": len([s for s in directional_sounds if s["sound_type"] == "elephant"]) if is_positive else 0
            }
        }
        
        return scene
    
    def generate_dataset(self) -> str:
        """Generate complete dataset and save as JSONL"""
        print(f"🎬 Generating {self.config.num_samples} scenes...")
        print(f"   - Positive samples: {self.config.positive_ratio * 100:.1f}%")
        print(f"   - Directional sounds per scene: {self.config.min_directional_sounds}-{self.config.max_directional_sounds}")
        print(f"   - Ambient sounds per scene: {self.config.min_ambient_sounds}-{self.config.max_ambient_sounds}")
        
        # Calculate number of positive samples
        num_positive = int(self.config.num_samples * self.config.positive_ratio)
        num_negative = self.config.num_samples - num_positive
        
        print(f"   - Positive scenes: {num_positive}")
        print(f"   - Negative scenes: {num_negative}")
        
        # Generate scenes
        scenes = []
        scene_id = 1
        
        # Generate positive samples
        for _ in tqdm(range(num_positive), desc="Generating positive scenes"):
            scene = self.generate_scene(is_positive=True, scene_id=scene_id)
            scenes.append(scene)
            scene_id += 1
        
        # Generate negative samples
        for _ in tqdm(range(num_negative), desc="Generating negative scenes"):
            scene = self.generate_scene(is_positive=False, scene_id=scene_id)
            scenes.append(scene)
            scene_id += 1
        
        # Shuffle scenes
        random.shuffle(scenes)
        
        # Save as JSONL
        jsonl_path = self.output_dir / f"generated_scenes_{self.config.num_samples}.jsonl"
        with open(jsonl_path, 'w') as f:
            for scene in scenes:
                f.write(json.dumps(scene) + '\n')
        
        print(f"✅ Generated {len(scenes)} scenes saved to: {jsonl_path}")
        
        # Generate summary statistics
        self._generate_summary(scenes, jsonl_path.parent / f"summary_{self.config.num_samples}.json")
        
        return str(jsonl_path)
    
    def _generate_summary(self, scenes: List[Dict], summary_path: Path):
        """Generate summary statistics for the dataset"""
        summary = {
            "total_scenes": len(scenes),
            "positive_scenes": len([s for s in scenes if s["metadata"]["is_positive"]]),
            "negative_scenes": len([s for s in scenes if not s["metadata"]["is_positive"]]),
            "sound_type_distribution": {},
            "height_distribution": {},
            "distance_distribution": {},
            "ambient_type_distribution": {}
        }
        
        # Analyze sound types and positions
        for scene in scenes:
            for sound in scene["sounds"]:
                sound_type = sound["sound_type"]
                summary["sound_type_distribution"][sound_type] = summary["sound_type_distribution"].get(sound_type, 0) + 1
                
                # Height analysis
                height = sound["position"][2]
                height_bin = f"{int(height//5)*5}-{int(height//5)*5+5}m"
                summary["height_distribution"][height_bin] = summary["height_distribution"].get(height_bin, 0) + 1
                
                # Distance analysis
                distance = sound["distance"]
                distance_bin = f"{int(distance//50)*50}-{int(distance//50)*50+50}m"
                summary["distance_distribution"][distance_bin] = summary["distance_distribution"].get(distance_bin, 0) + 1
            
            for ambient in scene["ambient"]:
                ambient_type = ambient["sound_type"]
                summary["ambient_type_distribution"][ambient_type] = summary["ambient_type_distribution"].get(ambient_type, 0) + 1
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary statistics saved to: {summary_path}")

class SceneRenderer:
    """Renders audio from JSONL scene configurations"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.simulator = SimpleSimulator(dataset_manager)
    
    def render_scenes_from_jsonl(self, jsonl_path: str, output_dir: str = None):
        """Render all scenes from a JSONL file to 4-channel audio"""
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        if output_dir is None:
            output_dir = jsonl_path.parent / "rendered_audio"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎵 Rendering scenes from: {jsonl_path}")
        print(f"🎯 Output directory: {output_dir}")
        
        # Count scenes
        with open(jsonl_path, 'r') as f:
            scenes = [json.loads(line) for line in f]
        
        print(f"📊 Found {len(scenes)} scenes to render")
        
        for i, scene_config in enumerate(tqdm(scenes, desc="Rendering scenes")):
            try:
                # Convert JSON config to SceneConfig object
                config = self._json_to_scene_config(scene_config)
                
                # Simulate scene
                mic_signals, metadata = self.simulator.simulate_scene(config)
                
                # Save results
                scene_output_dir = output_dir / scene_config["name"]
                self.simulator.save_results(mic_signals, metadata, str(scene_output_dir))
                
            except Exception as e:
                print(f"⚠️  Error rendering scene {i}: {e}")
                continue
        
        print(f"✅ Rendering complete! Audio files saved to: {output_dir}")
    
    def _json_to_scene_config(self, scene_json: Dict) -> SceneConfig:
        """Convert JSON scene configuration to SceneConfig object"""
        # Convert sounds
        sounds = []
        for sound_data in scene_json["sounds"]:
            sound = SoundPlacement(
                audio_file=sound_data["audio_file"],
                position=sound_data["position"],
                distance=sound_data["distance"],
                azimuth=sound_data["azimuth"],
                elevation=sound_data["elevation"],
                start_time=sound_data["start_time"],
                volume=sound_data["volume"],
                sound_type=sound_data["sound_type"]
            )
            sounds.append(sound)
        
        # Convert ambient sounds
        ambient = []
        for ambient_data in scene_json["ambient"]:
            ambient_sound = AmbientSound(
                sound_type=ambient_data["sound_type"],
                level_db=ambient_data["level_db"],
                audio_file=ambient_data["audio_file"]
            )
            ambient.append(ambient_sound)
        
        return SceneConfig(
            name=scene_json["name"],
            duration=scene_json["duration"],
            sounds=sounds,
            ambient=ambient,
            sample_rate=scene_json["sample_rate"]
        )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate elephant detection scene configurations")
    
    # Generation parameters
    parser.add_argument("--num-samples", type=int, default=100, help="Number of scenes to generate")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Ratio of positive samples (0.0-1.0)")
    parser.add_argument("--min-directional", type=int, default=10, help="Minimum directional sounds per scene")
    parser.add_argument("--max-directional", type=int, default=50, help="Maximum directional sounds per scene")
    parser.add_argument("--min-ambient", type=int, default=3, help="Minimum ambient sounds per scene")
    parser.add_argument("--max-ambient", type=int, default=5, help="Maximum ambient sounds per scene")
    parser.add_argument("--output-dir", type=str, default="generated_scenes", help="Output directory")
    
    # Execution options
    parser.add_argument("--render", action="store_true", help="Render audio from generated JSONL")
    parser.add_argument("--jsonl-path", type=str, help="Path to JSONL file to render (if --render)")
    parser.add_argument("--render-output", type=str, help="Output directory for rendered audio")
    
    # Scene parameters
    parser.add_argument("--duration", type=float, default=30.0, help="Scene duration in seconds")
    parser.add_argument("--min-distance", type=float, default=10.0, help="Minimum sound distance in meters")
    parser.add_argument("--max-distance", type=float, default=500.0, help="Maximum sound distance in meters")
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    print("🔍 Initializing dataset manager...")
    dataset_manager = DatasetManager()
    
    if args.render:
        # Render mode
        if not args.jsonl_path:
            print("❌ Error: --jsonl-path required when using --render")
            sys.exit(1)
        
        renderer = SceneRenderer(dataset_manager)
        renderer.render_scenes_from_jsonl(args.jsonl_path, args.render_output)
    else:
        # Generation mode
        config = GeneratorConfig(
            num_samples=args.num_samples,
            positive_ratio=args.positive_ratio,
            min_directional_sounds=args.min_directional,
            max_directional_sounds=args.max_directional,
            min_ambient_sounds=args.min_ambient,
            max_ambient_sounds=args.max_ambient,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            scene_duration=args.duration,
            output_dir=args.output_dir
        )
        
        generator = SceneGenerator(dataset_manager, config)
        jsonl_path = generator.generate_dataset()
        
        print(f"\n🎉 Dataset generation complete!")
        print(f"📁 JSONL file: {jsonl_path}")
        print(f"🎵 To render audio: python scene_generator.py --render --jsonl-path {jsonl_path}")

if __name__ == "__main__":
    main()