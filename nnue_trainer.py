import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import chess.pgn
import random
import os
import time
import datetime
from torch.utils.data import Dataset, DataLoader

# Periksa ketersediaan GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ChessFeatureEncoder:
    """Encoder fitur papan catur dengan fitur taktis dan strategis."""

    def __init__(self):
        # Definisi fitur dasar
        self.num_pieces = 12  # 6 jenis bidak × 2 warna
        self.num_squares = 64  # 64 kotak pada papan
        self.feature_dim = self.num_pieces * self.num_squares * 2  # Fitur untuk kedua pihak

        # Mapping pieces ke indeks
        self.piece_to_index = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

    def board_to_features(self, board):
        """Konversi board catur ke vektor fitur untuk NNUE."""
        # Inisialisasi vektor fitur sparse
        white_features = np.zeros(self.num_pieces * self.num_squares, dtype=np.float32)
        black_features = np.zeros(self.num_pieces * self.num_squares, dtype=np.float32)

        # Iterasi melalui semua bidak di papan
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_idx = self.piece_to_index[(piece.piece_type, piece.color)]

                # Fitur dari perspektif putih
                white_feat_idx = piece_idx * self.num_squares + square
                white_features[white_feat_idx] = 1.0

                # Fitur dari perspektif hitam (balik papan)
                black_square = chess.square_mirror(square)
                black_feat_idx = piece_idx * self.num_squares + black_square
                black_features[black_feat_idx] = 1.0

        # Gunakan perspektif yang sesuai berdasarkan giliran
        if board.turn == chess.WHITE:
            return torch.from_numpy(np.concatenate([white_features, black_features]))
        else:
            return torch.from_numpy(np.concatenate([black_features, white_features]))

    def calculate_attack_map(self, board):
        """Hitung peta serangan - kotak mana yang diserang oleh siapa."""
        # Array untuk menyimpan jumlah serangan per kotak untuk kedua warna
        white_attacks = np.zeros(64, dtype=np.float32)
        black_attacks = np.zeros(64, dtype=np.float32)

        # Periksa serangan untuk setiap bidak
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Dapatkan kotak yang diserang oleh bidak ini
            attacks = board.attacks(square)
            for attack_square in attacks:
                if piece.color == chess.WHITE:
                    white_attacks[attack_square] += 1
                else:
                    black_attacks[attack_square] += 1

        return white_attacks, black_attacks

class AttackCoordinationNNUE(nn.Module):
    """NNUE dengan kemampuan koordinasi serangan."""

    def __init__(self, feature_dim=768*2):
        super(AttackCoordinationNNUE, self).__init__()

        # Dimensi layer
        self.hidden_dim1 = 512
        self.hidden_dim2 = 256
        self.hidden_dim3 = 64

        # Feature transformer
        self.feature_transformer = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Target detector - memprediksi kotak yang menjadi target serangan
        self.target_detector = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim2, 64),  # Prediksi untuk 64 kotak
            nn.Sigmoid()  # Probabilitas 0-1 untuk setiap kotak
        )

        # Coordination evaluator - evaluasi tingkat koordinasi
        self.coordination_evaluator = nn.Sequential(
            nn.Linear(self.hidden_dim1 + 64, self.hidden_dim2),  # Hidden + target info
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim2, 1),
            nn.Sigmoid()  # Skor koordinasi 0-1
        )

        # Position evaluator - evaluasi posisi standard
        self.position_evaluator = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.ReLU(),
            nn.Linear(self.hidden_dim3, 1)
        )

        # Move evaluator - evaluasi gerakan dalam konteks koordinasi
        self.move_evaluator = nn.Sequential(
            nn.Linear(self.hidden_dim1 + 64, self.hidden_dim2),  # Hidden + target info
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim2, 1968)  # Skor untuk setiap gerakan (64*64 - invalid)
        )

        # Inisialisasi bobot
        self._init_weights()

        # Pindahkan ke GPU jika tersedia
        self.to(device)

        # Status koordinasi serangan
        self.attack_target = None
        self.coordination_level = 0.0
        self.preparation_moves = 0
        self.max_preparation_moves = 5
        self.coordination_threshold = 0.7
        self.attack_history = []

        # Encoder fitur
        self.feature_encoder = ChessFeatureEncoder()

    def _init_weights(self):
        """Inisialisasi bobot dengan metode Kaiming He."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, prev_target=None, planning_mode=False):
        """Forward pass dengan dukungan mode koordinasi."""
        # Transform fitur dasar
        base_features = self.feature_transformer(x)

        # Deteksi target serangan
        attack_targets = self.target_detector(base_features)

        # Jika target sudah diberikan, gunakan itu
        if prev_target is not None:
            attack_targets = prev_target

        # Evaluasi posisi standar
        position_score = self.position_evaluator(base_features)
        position_score = 100.0 * torch.tanh(position_score/100.0)  # Scale ke centipawns

        # Jika dalam mode koordinasi
        if planning_mode or prev_target is not None:
            # Gabungkan fitur dengan target untuk konteks
            combined_features = torch.cat([base_features, attack_targets], dim=1)

            # Evaluasi koordinasi
            coordination_score = self.coordination_evaluator(combined_features)

            # Evaluasi gerakan untuk koordinasi
            move_scores = self.move_evaluator(combined_features)

            # Blend position score with coordination focus
            blended_score = position_score * (1 - coordination_score) + 150.0 * coordination_score

            return blended_score, attack_targets, coordination_score, move_scores
        else:
            # Mode evaluasi normal
            return position_score, attack_targets, torch.zeros(1, 1).to(device), torch.zeros(1, 1968).to(device)

    def set_attack_target(self, square_idx):
        """Tetapkan target serangan dan masuk ke mode koordinasi."""
        self.attack_target = square_idx
        self.coordination_level = 0.0
        self.preparation_moves = 0
        self.attack_history = []

        # One-hot encode target
        target_tensor = torch.zeros(1, 64).to(device)
        target_tensor[0, square_idx] = 1.0

        return target_tensor

    def clear_attack_plan(self):
        """Reset rencana serangan."""
        self.attack_target = None
        self.coordination_level = 0.0
        self.preparation_moves = 0
        self.attack_history = []

    def update_coordination(self, coordination_score):
        """Update level koordinasi."""
        # Simple moving average update
        if self.coordination_level == 0:
            self.coordination_level = coordination_score
        else:
            # Weighted update
            self.coordination_level = 0.3 * self.coordination_level + 0.7 * coordination_score

        self.preparation_moves += 1
        self.attack_history.append(coordination_score)

        return self.coordination_level

    def is_ready_to_attack(self):
        """Cek apakah koordinasi mencapai threshold untuk serangan."""
        # Threshold check
        if self.coordination_level >= self.coordination_threshold:
            return True

        # Trend analysis - koordinasi meningkat konsisten
        if len(self.attack_history) >= 3:
            # Periksa 3 langkah terakhir
            last_three = self.attack_history[-3:]
            if all(last_three[i] <= last_three[i+1] for i in range(len(last_three)-1)):
                # Tren meningkat
                if last_three[-1] >= 0.6:  # Lower threshold for consistent improvement
                    return True

        # Time constraint check - jika sudah terlalu lama
        if self.preparation_moves >= self.max_preparation_moves:
            if self.coordination_level >= 0.5:  # Lower threshold for timeout
                return True

        return False

    def should_abandon_plan(self):
        """Cek apakah rencana serangan harus dibatalkan."""
        # Jika koordinasi tidak meningkat
        if self.preparation_moves >= 3:
            if len(self.attack_history) >= 3:
                if self.attack_history[-1] < self.attack_history[-3]:
                    return True

        # Jika koordinasi sangat rendah
        if self.preparation_moves >= 3 and self.coordination_level < 0.3:
            return True

        # Jika sudah terlalu lama mempersiapkan
        if self.preparation_moves >= self.max_preparation_moves and self.coordination_level < 0.5:
            return True

        return False

    def evaluate_position(self, board):
        """Evaluasi posisi papan catur."""
        # Simpan status training
        was_training = self.training

        # Set model ke mode evaluasi (penting untuk BatchNorm)
        self.eval()

        try:
            with torch.no_grad():
                features = self.feature_encoder.board_to_features(board).unsqueeze(0).to(device)
                value, _, _, _ = self.forward(features)
                result = value.item()

            return result
        finally:
            # Kembalikan model ke mode sebelumnya jika perlu
            if was_training:
                self.train()

    def select_best_move(self, board, planning_mode=False):
        """Pilih gerakan terbaik dengan strategi koordinasi serangan."""
        # Simpan status training
        was_training = self.training

        # Set model ke mode evaluasi (penting untuk BatchNorm)
        self.eval()

        try:
            # Get legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None, {"error": "No legal moves"}

            # One move? Return immediately
            if len(legal_moves) == 1:
                return legal_moves[0], {"info": "Only one legal move"}

            # Encode board features
            features = self.feature_encoder.board_to_features(board).unsqueeze(0).to(device)

            with torch.no_grad():
                # Check for active attack plan
                target_tensor = None
                if self.attack_target is not None:
                    # Get one-hot encoded target
                    target_tensor = torch.zeros(1, 64).to(device)
                    target_tensor[0, self.attack_target] = 1.0

                    # Force planning mode if we have an active target
                    planning_mode = True

                # Forward pass with appropriate mode
                value, targets, coordination, move_scores = self.forward(
                    features, prev_target=target_tensor, planning_mode=planning_mode
                )

                # No active plan - detect potential targets
                if self.attack_target is None and planning_mode:
                    # Select highest probability target
                    target_idx = targets.argmax().item()
                    target_prob = targets[0, target_idx].item()

                    # If target is promising, set as plan
                    if target_prob > 0.6:  # Threshold
                        target_tensor = self.set_attack_target(target_idx)

                        # Re-evaluate with target
                        value, targets, coordination, move_scores = self.forward(
                            features, prev_target=target_tensor, planning_mode=True
                        )
                    else:
                        # No good target, use normal evaluation
                        print(f"No promising targets found. Best: {target_idx} with prob {target_prob:.4f}")
                        return self._select_normal_move(board, legal_moves, value)

                # Active attack plan
                if self.attack_target is not None:
                    coord_score = coordination.item()

                    # Update internal coordination tracking
                    self.update_coordination(coord_score)

                    # Check if ready to attack
                    if self.is_ready_to_attack():
                        print(f"Coordination threshold reached! Executing attack...")
                        move, info = self._select_attacking_move(board, legal_moves, self.attack_target, move_scores)

                        # Clear plan after attack
                        self.clear_attack_plan()
                        return move, info

                    # Check if should abandon plan
                    if self.should_abandon_plan():
                        print(f"Abandoning attack plan - insufficient coordination")
                        self.clear_attack_plan()
                        return self._select_normal_move(board, legal_moves, value)

                    # Continue with coordination moves
                    return self._select_coordination_move(board, legal_moves, move_scores)

                # No active plan and not in planning mode
                return self._select_normal_move(board, legal_moves, value)

        finally:
            # Kembalikan model ke mode sebelumnya jika perlu
            if was_training:
                self.train()

    def _select_coordination_move(self, board, legal_moves, move_scores):
        """Pilih gerakan koordinasi terbaik."""
        # Simpan status training
        was_training = self.training

        # Set model ke mode evaluasi
        self.eval()

        try:
            best_move = None
            best_score = float('-inf')
            move_infos = {}

            for move in legal_moves:
                # Get move index
                from_square = move.from_square
                to_square = move.to_square
                move_idx = from_square * 64 + to_square

                # Get score from network
                if move_idx < 1968:
                    score = move_scores[0, move_idx].item()

                    # Boost score for moves towards target
                    target_row, target_col = divmod(self.attack_target, 8)
                    to_row, to_col = divmod(to_square, 8)

                    # Distance to target
                    distance = max(abs(to_row - target_row), abs(to_col - target_col))
                    distance_factor = 1.0 - (distance / 8.0)  # 1.0 for same square, 0 for furthest

                    # Adjusted score with bias towards target
                    adjusted_score = score + (distance_factor * 0.2)

                    move_infos[move.uci()] = {
                        "raw_score": score,
                        "distance_factor": distance_factor,
                        "adjusted_score": adjusted_score
                    }

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_move = move

            if best_move:
                info = {
                    "type": "coordination",
                    "move": best_move.uci(),
                    "score": best_score,
                    "target": self.attack_target,
                    "coordination_level": self.coordination_level,
                    "preparation_move": self.preparation_moves,
                    "move_details": move_infos[best_move.uci()]
                }
                print(f"Selected coordination move: {best_move.uci()} with score {best_score:.4f}")
                return best_move, info
            else:
                # Fallback to normal move
                print("No good coordination move found, falling back to normal evaluation")
                return self._select_normal_move(board, legal_moves, None)

        finally:
            # Kembalikan model ke mode sebelumnya
            if was_training:
                self.train()

    def _select_attacking_move(self, board, legal_moves, target_square, move_scores=None):
        """Pilih gerakan serangan terbaik ke target."""
        # Simpan status training
        was_training = self.training

        # Set model ke mode evaluasi
        self.eval()

        try:
            direct_attacks = []
            nearby_moves = []

            for move in legal_moves:
                # Check if move directly attacks target
                if move.to_square == target_square:
                    direct_attacks.append(move)

                # Check if move is near target
                to_row, to_col = divmod(move.to_square, 8)
                target_row, target_col = divmod(target_square, 8)

                # Within 2 squares of target
                if max(abs(to_row - target_row), abs(to_col - target_col)) <= 2:
                    nearby_moves.append((move, max(abs(to_row - target_row), abs(to_col - target_col))))

            # Prioritize direct attacks
            if direct_attacks:
                best_attack = direct_attacks[0]
                if len(direct_attacks) > 1 and move_scores is not None:
                    # Choose highest scored attack
                    best_score = float('-inf')
                    for move in direct_attacks:
                        move_idx = move.from_square * 64 + move.to_square
                        if move_idx < 1968:
                            score = move_scores[0, move_idx].item()
                            if score > best_score:
                                best_score = score
                                best_attack = move

                info = {
                    "type": "attack_execution",
                    "move": best_attack.uci(),
                    "target": target_square,
                    "direct_attack": True
                }
                print(f"Executing direct attack move: {best_attack.uci()}")
                return best_attack, info

            # No direct attacks, choose closest move to target
            if nearby_moves:
                # Sort by distance
                nearby_moves.sort(key=lambda x: x[1])

                best_move, distance = nearby_moves[0]

                info = {
                    "type": "attack_approach",
                    "move": best_move.uci(),
                    "target": target_square,
                    "distance": distance
                }
                print(f"Executing approach move: {best_move.uci()} (distance: {distance})")
                return best_move, info

            # Fallback to normal move
            print("No attack or approach moves found, falling back to normal evaluation")
            return self._select_normal_move(board, legal_moves, None)

        finally:
            # Kembalikan model ke mode sebelumnya
            if was_training:
                self.train()

    def _select_normal_move(self, board, legal_moves, value=None):
        """Pilih gerakan berdasarkan evaluasi posisi normal."""
        # Simpan status training
        was_training = self.training

        # Set model ke mode evaluasi
        self.eval()

        try:
            best_move = None
            best_eval = float('-inf')
            evals = {}

            for move in legal_moves:
                # Execute move
                board_copy = board.copy()
                board_copy.push(move)

                # Evaluate new position
                with torch.no_grad():
                    try:
                        move_features = self.feature_encoder.board_to_features(board_copy).unsqueeze(0).to(device)
                        new_value, _, _, _ = self.forward(move_features)
                        eval_score = new_value.item()
                    except Exception as e:
                        print(f"Error evaluating move {move.uci()}: {e}")
                        eval_score = -10000  # Penalize error moves

                    evals[move.uci()] = eval_score

                    if eval_score > best_eval:
                        best_eval = eval_score
                        best_move = move

            if best_move:
                info = {
                    "type": "normal",
                    "move": best_move.uci(),
                    "score": best_eval,
                    "all_evals": evals
                }
                print(f"Selected normal move: {best_move.uci()} with score {best_eval:.2f}")
                return best_move, info
            else:
                # Should never happen unless no legal moves
                return legal_moves[0], {"type": "fallback"}

        finally:
            # Kembalikan model ke mode sebelumnya
            if was_training:
                self.train()

    def save_model(self, path):
        """Save model to file."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'attack_coordination_settings': {
                'max_preparation_moves': self.max_preparation_moves,
                'coordination_threshold': self.coordination_threshold
            }
        }

        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path, map_location=None):
        """Load model from file."""
        if map_location is None:
            map_location = device

        try:
            checkpoint = torch.load(path, map_location=map_location)

            # Load coordination settings
            if 'attack_coordination_settings' in checkpoint:
                settings = checkpoint['attack_coordination_settings']
                if 'max_preparation_moves' in settings:
                    self.max_preparation_moves = settings['max_preparation_moves']
                if 'coordination_threshold' in settings:
                    self.coordination_threshold = settings['coordination_threshold']

            # Load state dict
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class ChessDataset(Dataset):
    """Dataset for training attack coordination NNUE."""

    def __init__(self, positions_data, feature_encoder):
        """
        Initialize dataset.

        Args:
            positions_data: List of tuples (board, targets)
                where targets is (value, target_squares, coordination, move_scores)
            feature_encoder: Feature encoder to convert boards to vectors
        """
        self.positions_data = positions_data
        self.feature_encoder = feature_encoder

    def __len__(self):
        return len(self.positions_data)

    def __getitem__(self, idx):
        board, targets = self.positions_data[idx]

        # Encode features
        features = self.feature_encoder.board_to_features(board)

        return features, targets

def prepare_targets(board, evaluation, best_move, attack_target=None, coordination_score=None):
    """
    Prepare training targets for a position.

    Args:
        board: chess.Board position
        evaluation: Float, position evaluation in centipawns
        best_move: chess.Move, best move in position
        attack_target: Optional int, target square for attack
        coordination_score: Optional float, coordination score
    """
    # 1. Value target
    value_target = torch.tensor([evaluation / 100.0], dtype=torch.float32)

    # 2. Target squares
    target_squares = torch.zeros(64, dtype=torch.float32)

    if attack_target is not None:
        # Use provided target
        target_squares[attack_target] = 1.0
    elif best_move:
        # Use destination of best move as target
        target_square = best_move.to_square
        target_squares[target_square] = 1.0

    # 3. Coordination target
    if coordination_score is not None:
        coordination_target = torch.tensor([coordination_score], dtype=torch.float32)
    else:
        coordination_target = torch.tensor([0.5], dtype=torch.float32)

    # 4. Move scores
    move_scores = torch.zeros(1968, dtype=torch.float32)

    if best_move:
        # Set score for best move
        from_square = best_move.from_square
        to_square = best_move.to_square
        move_idx = from_square * 64 + to_square

        if move_idx < 1968:
            move_scores[move_idx] = 1.0

    return value_target, target_squares, coordination_target, move_scores

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """Train the model."""
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Loss functions
    value_criterion = nn.MSELoss()
    target_criterion = nn.BCELoss()
    coordination_criterion = nn.MSELoss()
    move_criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        batch_count = 0

        for features, targets in train_loader:
            # Move data to device
            features = features.to(device)
            value_targets, target_squares, coordination_targets, move_targets = [t.to(device) for t in targets]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass in planning mode
            value_output, target_output, coordination_output, move_output = model(
                features, prev_target=target_squares, planning_mode=True
            )

            # Calculate losses
            value_loss = value_criterion(value_output, value_targets)
            target_loss = target_criterion(target_output, target_squares)
            coordination_loss = coordination_criterion(coordination_output, coordination_targets)
            move_loss = move_criterion(move_output, move_targets)

            # Combined loss
            combined_loss = 0.4 * value_loss + 0.2 * target_loss + 0.2 * coordination_loss + 0.2 * move_loss

            # Backward and optimize
            combined_loss.backward()
            optimizer.step()

            total_train_loss += combined_loss.item()
            batch_count += 1

        avg_train_loss = total_train_loss / batch_count

        # Validation phase
        model.eval()
        total_val_loss = 0
        batch_count = 0

        with torch.no_grad():
            for features, targets in val_loader:
                # Move data to device
                features = features.to(device)
                value_targets, target_squares, coordination_targets, move_targets = [t.to(device) for t in targets]

                # Forward pass in planning mode
                value_output, target_output, coordination_output, move_output = model(
                    features, prev_target=target_squares, planning_mode=True
                )

                # Calculate losses
                value_loss = value_criterion(value_output, value_targets)
                target_loss = target_criterion(target_output, target_squares)
                coordination_loss = coordination_criterion(coordination_output, coordination_targets)
                move_loss = move_criterion(move_output, move_targets)

                # Combined loss
                combined_loss = 0.4 * value_loss + 0.2 * target_loss + 0.2 * coordination_loss + 0.2 * move_loss

                total_val_loss += combined_loss.item()
                batch_count += 1

        avg_val_loss = total_val_loss / batch_count

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save if best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_model(f"attack_coord_best.pt")
            print("Saved new best model!")

    # Save final model
    model.save_model(f"attack_coord_final.pt")
    print("Training complete!")

def extract_training_data(pgn_file, num_positions=1000):
    """Extract training data from a PGN file."""
    import chess.pgn

    data = []

    with open(pgn_file) as f:
        game_count = 0
        positions_found = 0

        while positions_found < num_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_count += 1

            # Skip short games
            if game.end().ply() < 20:
                continue

            # Process this game
            board = game.board()
            node = game

            while node.variations and positions_found < num_positions:
                next_node = node.variations[0]
                move = next_node.move

                # Skip early game positions
                if board.fullmove_number < 10:
                    board.push(move)
                    node = next_node
                    continue

                # Create targets for this position
                targets = prepare_targets(
                    board,
                    0.0,  # No evaluation available, use 0.0
                    move
                )

                # Add to dataset
                data.append((board.copy(), targets))
                positions_found += 1

                # Make the move
                board.push(move)
                node = next_node

            # Status update
            if game_count % 10 == 0:
                print(f"Processed {game_count} games, found {positions_found} positions")

    print(f"Extracted {len(data)} positions from {game_count} games")
    return data

def create_attack_coordination_nnue():
    """Create and train a new Attack Coordination NNUE."""
    # Create model
    print("Creating Attack Coordination NNUE model...")
    model = AttackCoordinationNNUE()

    # Ask for PGN file
    pgn_file = input("Enter path to PGN file: ")
    if not os.path.exists(pgn_file):
        print("PGN file not found!")
        return

    # Extract training data
    print("Extracting training data...")
    num_positions = int(input("Number of positions to extract (default: 5000): ") or "5000")
    data = extract_training_data(pgn_file, num_positions)

    # Split into train and validation
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create datasets
    print("Creating datasets...")
    train_dataset = ChessDataset(train_data, model.feature_encoder)
    val_dataset = ChessDataset(val_data, model.feature_encoder)

    # Create dataloaders
    batch_size = int(input("Batch size (default: 256): ") or "256")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train model
    print("Training model...")
    num_epochs = int(input("Number of epochs (default: 10): ") or "10")
    lr = float(input("Learning rate (default: 0.001): ") or "0.001")

    train_model(model, train_loader, val_loader, num_epochs, lr)

    print("Done!")
    return model

def test_model(model_path=None):
    """Test a trained model."""
    # Load model
    if model_path is None:
        model_path = input("Enter path to model: ")

    model = AttackCoordinationNNUE()
    success = model.load_model(model_path)

    if not success:
        print("Failed to load model!")
        return

    # Ensure model is in evaluation mode
    model.eval()

    # Create a new board
    board = chess.Board()

    # Set up test position
    fen = input("Enter FEN (or press Enter for starting position): ")
    if fen:
        try:
            board.set_fen(fen)
        except Exception as e:
            print(f"Invalid FEN: {e}")
            return

    # Show board
    print(board)

    # Analyze position
    print("\nAnalyzing position...")

    # Normal evaluation
    value = model.evaluate_position(board)
    print(f"Position evaluation: {value:.2f} centipawns")

    # Find target
    features = model.feature_encoder.board_to_features(board).unsqueeze(0).to(device)
    with torch.no_grad():
        _, targets, _, _ = model.forward(features)
        target_idx = targets.argmax().item()
        target_prob = targets[0, target_idx].item()

        target_row, target_col = divmod(target_idx, 8)
        target_square = chess.square(target_col, target_row)
        target_name = chess.square_name(target_square)

        print(f"Detected attack target: {target_name} (square {target_idx}) with probability {target_prob:.4f}")

    # Get best move with attack coordination
    model.clear_attack_plan()
    move, info = model.select_best_move(board, planning_mode=True)

    if move:
        print(f"\nBest move: {move.uci()}")
        print(f"Move type: {info.get('type', 'normal')}")

        if 'coordination_level' in info:
            print(f"Coordination level: {info['coordination_level']:.4f}")
    else:
        print("No legal moves!")

if __name__ == "__main__":
    print("Attack Coordination NNUE System")
    print("-" * 30)

    # Menu
    print("\nChoose action:")
    print("1. Create and train new model")
    print("2. Test existing model")

    choice = input("Enter choice (1-2): ")

    if choice == "1":
        create_attack_coordination_nnue()
    elif choice == "2":
        test_model()
    else:
        print("Invalid choice!")
