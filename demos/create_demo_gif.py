"""
Script pour créer un GIF de démonstration sans dépendre de l'environnement.

Ce script crée un GIF avec des frames annotées pour démontrer le système.
"""

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_demo_frame(step: int, action: int, instruction: str, reward: float) -> np.ndarray:
    """Créer une frame de démonstration."""
    # Créer une image de base (simulation d'environnement) - TAILLE AUGMENTEE
    width, height = 640, 480
    img = Image.new('RGB', (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Dessiner un environnement simple (grille) - GRILLE PLUS GRANDE
    grid_size = 10
    cell_size = min(width, height) // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = i * cell_size
            y = j * cell_size
            if (i + j) % 2 == 0:
                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=(100, 100, 100))
    
    # Dessiner un "goal" vert
    goal_x, goal_y = 6 * cell_size, 6 * cell_size
    draw.ellipse([goal_x, goal_y, goal_x + cell_size, goal_y + cell_size], fill=(0, 255, 0))
    
    # Dessiner un "agent" rouge (position qui change)
    agent_x = (step % 4) * cell_size
    agent_y = (step // 4) * cell_size
    draw.ellipse([agent_x + 5, agent_y + 5, agent_x + cell_size - 5, agent_y + cell_size - 5], fill=(255, 0, 0))
    
    # Annotations - POLICES PLUS GRANDES
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except:
        # Si pas de police système, utiliser une taille par défaut plus grande
        try:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = None
            font_small = None
    
    # Instruction en haut - ZONE PLUS GRANDE
    instruction_text = f"Instruction: {instruction}"
    draw.rectangle([(5, 5), (width - 5, 50)], fill=(0, 0, 0, 200))
    draw.text((10, 15), instruction_text, fill=(255, 255, 255), font=font_large)
    
    # Informations en bas - ZONE PLUS GRANDE
    action_names = {0: "Turn Left", 1: "Turn Right", 2: "Move Forward", 3: "Pick Up/Drop"}
    action_name = action_names.get(action, f"Action {action}")
    info_text = f"Step: {step} | Action: {action_name} ({action}) | Reward: {reward:.2f}"
    
    if font_small:
        text_bbox = draw.textbbox((0, 0), info_text, font=font_small)
        text_height = text_bbox[3] - text_bbox[1]
    else:
        text_height = 25
    y_pos = height - text_height - 15
    
    draw.rectangle([(5, y_pos - 10), (width - 5, height - 5)], fill=(0, 0, 0, 200))
    draw.text((10, y_pos), info_text, fill=(255, 255, 255), font=font_small)
    
    return np.array(img)


def main():
    """Créer le GIF de démonstration."""
    instruction = "Navigate to the green goal"
    output_path = "demos/demonstration.gif"
    
    # Créer le dossier si nécessaire
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("Création du GIF de démonstration...")
    print(f"Instruction: {instruction}")
    print(f"Sortie: {output_path}")
    
    # Générer des frames
    frames = []
    num_steps = 30  # Plus de frames pour une démonstration plus longue
    
    for step in range(num_steps):
        # Simuler des actions
        action = step % 4  # Cycle through actions
        reward = 0.1 * step if step < 25 else 1.0  # Reward augmente puis succès
        
        frame = create_demo_frame(step, action, instruction, reward)
        frames.append(frame)
        
        print(f"Frame {step + 1}/{num_steps} créée")
    
    # Créer le GIF - VITESSE RALENTIE (1.5 secondes par frame)
    print("\nGénération du GIF...")
    imageio.mimsave(output_path, frames, duration=1.5)  # 1.5 secondes par frame (plus lent)
    
    print(f"\nGIF cree avec succes: {output_path}")
    print(f"  - {num_steps} frames")
    print(f"  - Duree: {num_steps * 1.5:.1f} secondes")
    print(f"  - Taille: 640x480 pixels")
    print(f"  - Format: GIF anime")


if __name__ == "__main__":
    main()

