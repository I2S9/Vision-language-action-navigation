"""
Script pour créer un GIF de démonstration sans dépendre de l'environnement.

Ce script crée un GIF avec des frames annotées pour démontrer le système.
"""

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_demo_frame(step: int, action: int, instruction: str, reward: float, 
                     agent_pos: tuple, agent_orientation: int, success: bool = False) -> np.ndarray:
    """Créer une frame de démonstration."""
    # Créer une image de base (simulation d'environnement) - TAILLE AUGMENTEE
    width, height = 640, 480
    img = Image.new('RGB', (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Dessiner un environnement simple (grille) - GRILLE QUI COUVRE TOUTE L'IMAGE
    # Calculer la taille des cellules pour couvrir toute l'image
    grid_cols = 12  # Plus de colonnes pour couvrir toute la largeur
    grid_rows = 10
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    
    # Dessiner la grille sur toute l'image
    for i in range(grid_cols):
        for j in range(grid_rows):
            x = i * cell_width
            y = j * cell_height
            if (i + j) % 2 == 0:
                draw.rectangle([x, y, x + cell_width, y + cell_height], fill=(100, 100, 100))
    
    # Dessiner un "goal" vert (position fixe)
    # Utiliser une grille 10x10 pour le positionnement (comme avant)
    grid_size = 10
    cell_size = min(width, height) // grid_size
    goal_grid_x, goal_grid_y = 6, 6
    goal_x = goal_grid_x * cell_size
    goal_y = goal_grid_y * cell_size
    draw.ellipse([goal_x, goal_y, goal_x + cell_size, goal_y + cell_size], fill=(0, 255, 0))
    
    # Dessiner l'agent avec position et orientation
    agent_x, agent_y = agent_pos
    agent_pixel_x = agent_x * cell_size + cell_size // 2
    agent_pixel_y = agent_y * cell_size + cell_size // 2
    
    # Robot bleu (carré simple) pour l'agent - AGRANDI
    agent_size = cell_size // 1.8  # Robot plus grand
    agent_half = agent_size // 2
    
    # Dessiner un carré simple pour le robot (sans contour)
    robot_box = [
        agent_pixel_x - agent_half,
        agent_pixel_y - agent_half,
        agent_pixel_x + agent_half,
        agent_pixel_y + agent_half
    ]
    draw.rectangle(robot_box, fill=(0, 100, 255))
    
    # Flèche pour indiquer l'orientation (0=North, 1=East, 2=South, 3=West)
    arrow_length = agent_half + 8
    arrow_end_x = agent_pixel_x
    arrow_end_y = agent_pixel_y
    
    if agent_orientation == 0:  # North (up)
        arrow_end_y = agent_pixel_y - arrow_length
    elif agent_orientation == 1:  # East (right)
        arrow_end_x = agent_pixel_x + arrow_length
    elif agent_orientation == 2:  # South (down)
        arrow_end_y = agent_pixel_y + arrow_length
    elif agent_orientation == 3:  # West (left)
        arrow_end_x = agent_pixel_x - arrow_length
    
    # Dessiner la flèche (rouge)
    draw.line([agent_pixel_x, agent_pixel_y, arrow_end_x, arrow_end_y], 
              fill=(255, 0, 0), width=3)
    
    # Pointe de la flèche (triangle rouge)
    arrow_head_size = 5
    if agent_orientation == 0:  # North
        points = [(arrow_end_x, arrow_end_y), 
                  (arrow_end_x - arrow_head_size, arrow_end_y + arrow_head_size),
                  (arrow_end_x + arrow_head_size, arrow_end_y + arrow_head_size)]
    elif agent_orientation == 1:  # East
        points = [(arrow_end_x, arrow_end_y),
                  (arrow_end_x - arrow_head_size, arrow_end_y - arrow_head_size),
                  (arrow_end_x - arrow_head_size, arrow_end_y + arrow_head_size)]
    elif agent_orientation == 2:  # South
        points = [(arrow_end_x, arrow_end_y),
                  (arrow_end_x - arrow_head_size, arrow_end_y - arrow_head_size),
                  (arrow_end_x + arrow_head_size, arrow_end_y - arrow_head_size)]
    else:  # West
        points = [(arrow_end_x, arrow_end_y),
                  (arrow_end_x + arrow_head_size, arrow_end_y - arrow_head_size),
                  (arrow_end_x + arrow_head_size, arrow_end_y + arrow_head_size)]
    
    draw.polygon(points, fill=(255, 0, 0))
    
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
    
    # Instruction en haut - BANDEAU AUGMENTE - CENTREE HORIZONTALEMENT ET VERTICALEMENT
    instruction_text = f"Instruction: {instruction}"
    bandeau_height = 50  # Hauteur augmentée du bandeau
    draw.rectangle([(5, 5), (width - 5, bandeau_height)], fill=(0, 0, 0, 200))
    
    # Centrer le texte horizontalement et verticalement
    if font_large:
        text_bbox = draw.textbbox((0, 0), instruction_text, font=font_large)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else:
        text_width = len(instruction_text) * 10  # Estimation
        text_height = 20
    
    # Centrage horizontal
    text_x = (width - text_width) // 2
    # Centrage vertical dans le bandeau (5 est le padding en haut, bandeau_height est la hauteur totale)
    text_y = 5 + (bandeau_height - 5 - text_height) // 2
    draw.text((text_x, text_y), instruction_text, fill=(255, 255, 255), font=font_large)
    
    # Informations en bas - ZONE PLUS GRANDE avec progression - CENTREE
    action_names = {0: "Turn Left", 1: "Turn Right", 2: "Move Forward", 3: "Pick Up/Drop"}
    action_name = action_names.get(action, f"Action {action}")
    
    # Afficher la progression
    progress_text = f"Step {step}"
    info_text = f"{progress_text} | Action: {action_name} | Reward: {reward:.2f}"
    
    if font_small:
        text_bbox = draw.textbbox((0, 0), info_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else:
        text_width = len(info_text) * 8  # Estimation
        text_height = 25
    
    y_pos = height - text_height - 15
    
    draw.rectangle([(5, y_pos - 10), (width - 5, height - 5)], fill=(0, 0, 0, 200))
    
    # Centrer le texte
    text_x = (width - text_width) // 2
    draw.text((text_x, y_pos), info_text, fill=(255, 255, 255), font=font_small)
    
    # Afficher "SUCCESS" si l'agent a atteint le goal - CENTRE HORIZONTALEMENT ET VERTICALEMENT
    if success:
        success_text = "SUCCESS"
        if font_large:
            success_bbox = draw.textbbox((0, 0), success_text, font=font_large)
            success_width = success_bbox[2] - success_bbox[0]
            success_height = success_bbox[3] - success_bbox[1]
        else:
            success_width = 200
            success_height = 30
        
        # Dimensions du rectangle vert
        rect_padding = 10
        rect_width = success_width + 2 * rect_padding
        rect_height = success_height + 2 * rect_padding
        
        # Centrer le rectangle horizontalement et verticalement
        rect_x = (width - rect_width) // 2
        rect_y = (height - rect_height) // 2
        
        # Fond pour le message de succès (sans contour)
        draw.rectangle([rect_x, rect_y, 
                       rect_x + rect_width, rect_y + rect_height],
                      fill=(0, 150, 0))
        
        # Centrer le texte dans le rectangle (horizontalement et verticalement)
        # Calculer la position exacte pour un centrage parfait
        text_x = rect_x + rect_padding + (rect_width - 2 * rect_padding - success_width) // 2
        text_y = rect_y + rect_padding + (rect_height - 2 * rect_padding - success_height) // 2
        draw.text((text_x, text_y), success_text, fill=(255, 255, 255), font=font_large)
    
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
    
    # Simuler une trajectoire réussie vers le goal
    # Position initiale de l'agent (plus proche pour GIF court)
    agent_x, agent_y = 2, 2
    agent_orientation = 1  # 0=North, 1=East, 2=South, 3=West
    goal_x, goal_y = 6, 6  # Goal plus proche pour trajectoire courte
    
    # Générer des frames avec une trajectoire réaliste vers le goal
    frames = []
    step = 0
    max_steps = 15  # Maximum de steps pour atteindre le goal
    
    while step < max_steps:
        # Vérifier si on a atteint le goal
        if agent_x == goal_x and agent_y == goal_y:
            # Atteint le goal - afficher SUCCESS
            action = 2  # Move forward (arrive)
            reward = 1.0
            success = True
            
            # Créer la frame avec SUCCESS
            frame = create_demo_frame(
                step=step,
                action=action,
                instruction=instruction,
                reward=reward,
                agent_pos=(agent_x, agent_y),
                agent_orientation=agent_orientation,
                success=True
            )
            frames.append(frame)
            print(f"Frame {step + 1} créée - SUCCESS! Agent: ({agent_x}, {agent_y}), Reward: {reward:.2f}")
            
            # Ajouter 2 frames supplémentaires avec le message SUCCESS pour bien le voir
            # Garder le même step pour ces frames (ne pas incrémenter)
            for i in range(2):
                frame = create_demo_frame(
                    step=step,  # Garder le même step que la frame de succès
                    action=action,
                    instruction=instruction,
                    reward=reward,
                    agent_pos=(agent_x, agent_y),
                    agent_orientation=agent_orientation,
                    success=True
                )
                frames.append(frame)
            break
        
        # Se diriger vers le goal
        dx = goal_x - agent_x
        dy = goal_y - agent_y
        
        # Choisir l'action pour se rapprocher du goal (priorité à la direction la plus éloignée)
        if dx != 0:
            # Priorité à se déplacer horizontalement
            target_orientation = 1 if dx > 0 else 3
        elif dy != 0:
            # Puis verticalement
            target_orientation = 2 if dy > 0 else 0
        else:
            # Si dx == 0 et dy == 0, on est arrivé (ne devrait pas arriver ici)
            target_orientation = agent_orientation
        
        if agent_orientation == target_orientation:
            action = 2  # Move forward
            if agent_orientation == 1:  # East
                agent_x = min(9, agent_x + 1)
            elif agent_orientation == 3:  # West
                agent_x = max(0, agent_x - 1)
            elif agent_orientation == 2:  # South
                agent_y = min(9, agent_y + 1)
            else:  # North
                agent_y = max(0, agent_y - 1)
        else:
            # Tourner vers la bonne direction
            # Calculer le chemin le plus court pour tourner
            diff = (target_orientation - agent_orientation) % 4
            if diff == 1 or diff == 3:
                # Tourner dans la direction la plus courte
                if diff == 1:
                    action = 1  # Turn right
                    agent_orientation = (agent_orientation + 1) % 4
                else:
                    action = 0  # Turn left
                    agent_orientation = (agent_orientation - 1) % 4
            else:
                # Différence de 2, tourner à droite par défaut
                action = 1  # Turn right
                agent_orientation = (agent_orientation + 1) % 4
        
        # Calculer le reward (augmente quand on se rapproche)
        new_distance = ((agent_x - goal_x)**2 + (agent_y - goal_y)**2)**0.5
        reward = max(0.1, 1.0 - new_distance * 0.15)
        success = False
        
        # Créer la frame
        frame = create_demo_frame(
            step=step,
            action=action,
            instruction=instruction,
            reward=reward,
            agent_pos=(agent_x, agent_y),
            agent_orientation=agent_orientation,
            success=success
        )
        frames.append(frame)
        
        print(f"Frame {step + 1} créée - Agent: ({agent_x}, {agent_y}), Orientation: {agent_orientation}, Reward: {reward:.2f}, Distance: {new_distance:.2f}")
        step += 1
    
    # Créer le GIF - DUREE TOTALE 5-10 SECONDES
    # Ajuster la durée pour avoir entre 5 et 10 secondes
    total_frames = len(frames)
    target_duration = 8.0  # Cible: 8 secondes
    duration_per_frame = target_duration / total_frames if total_frames > 0 else 0.8
    duration_per_frame = max(0.5, min(1.0, duration_per_frame))  # Entre 0.5 et 1.0 seconde par frame
    
    print(f"\nGénération du GIF...")
    imageio.mimsave(output_path, frames, duration=duration_per_frame)
    
    print(f"\nGIF cree avec succes: {output_path}")
    print(f"  - {total_frames} frames")
    print(f"  - Duree: {total_frames * duration_per_frame:.1f} secondes")
    print(f"  - Taille: 640x480 pixels")
    print(f"  - Format: GIF anime")
    print(f"  - Trajectoire: Reussie (agent atteint le goal)")


if __name__ == "__main__":
    main()

