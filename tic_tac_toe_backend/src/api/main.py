from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Literal
from passlib.context import CryptContext
from jose import jwt, JWTError
from uuid import uuid4
import time

SECRET_KEY = "replace_this_with_proper_secret"  # Should be set via .env in real deployment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 3600

app = FastAPI(
    title="Tic-Tac-Toe API",
    description="Backend for a full-stack tic-tac-toe game with user registration, authentication, game logic, and leaderboard.",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "User registration and authentication"},
        {"name": "game", "description": "Game logic and state endpoints"},
        {"name": "leaderboard", "description": "Leaderboard display"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: int = ACCESS_TOKEN_EXPIRE_SECONDS):
    to_encode = data.copy()
    expire = int(time.time()) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# --- In-memory stores (replace with database for production) ---
users_db: Dict[str, dict] = {}  # username/email -> User dict
games_db: Dict[str, dict] = {}  # game_id (str) -> Game dict
moves_db: Dict[str, List[dict]] = {}  # game_id -> list of Move dicts

# --- Models ---

# PUBLIC_INTERFACE
class UserRegisterRequest(BaseModel):
    """User registration input."""
    username: str = Field(..., min_length=3, max_length=32, description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password")

# PUBLIC_INTERFACE
class UserLoginRequest(BaseModel):
    """User login input."""
    username: str
    password: str

# PUBLIC_INTERFACE
class Token(BaseModel):
    """Access token returned after authentication."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")

# PUBLIC_INTERFACE
class User(BaseModel):
    """User persistent model."""
    id: str = Field(..., description="Unique user id (UUID string)")
    username: str
    email: EmailStr
    hashed_password: str
    games_played: int = 0
    games_won: int = 0
    games_lost: int = 0
    games_tied: int = 0

# PUBLIC_INTERFACE
class OpponentType(BaseModel):
    type: Literal['ai', 'human'] = Field(..., description="Play against an AI or a human")

# PUBLIC_INTERFACE
class GameStartRequest(BaseModel):
    opponent: Literal['ai', 'human'] = Field(..., description="Choose AI or specify username for human")
    opponent_username: Optional[str] = Field(None, description="If human, provide opponent's username.")

class GameStatus(str):
    ONGOING = 'ongoing'
    X_WON = 'x_won'
    O_WON = 'o_won'
    TIE = 'tie'

# PUBLIC_INTERFACE
class Game(BaseModel):
    id: str
    player_x: str
    player_o: str
    board: List[List[str]] = Field(default_factory=lambda: [["" for _ in range(3)] for _ in range(3)], description="3x3 grid")
    current_turn: str = 'X'
    status: str = GameStatus.ONGOING
    winner: Optional[str] = None

# PUBLIC_INTERFACE
class MoveRequest(BaseModel):
    game_id: str
    row: int = Field(..., ge=0, le=2)
    col: int = Field(..., ge=0, le=2)

# PUBLIC_INTERFACE
class Move(BaseModel):
    player: str
    row: int
    col: int
    symbol: str  # 'X' or 'O'
    timestamp: float

# PUBLIC_INTERFACE
class GameStateResponse(BaseModel):
    game: Game
    moves: List[Move]

# PUBLIC_INTERFACE
class LeaderboardEntry(BaseModel):
    username: str
    games_won: int
    games_played: int
    games_lost: int
    games_tied: int

# PUBLIC_INTERFACE
class LeaderboardResponse(BaseModel):
    entries: List[LeaderboardEntry]

# --- Auth Logic ---

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    payload = decode_access_token(token)
    if not payload or 'sub' not in payload:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    username = payload['sub']
    user = users_db.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user)

# --- Game Logic ---

def check_winner(board: List[List[str]]) -> Optional[str]:
    # Horizontal, vertical, diagonal checks
    for i in range(3):
        if all(board[i][j] == 'X' for j in range(3)): return 'X'
        if all(board[i][j] == 'O' for j in range(3)): return 'O'
        if all(board[j][i] == 'X' for j in range(3)): return 'X'
        if all(board[j][i] == 'O' for j in range(3)): return 'O'
    # Diagonals
    if all(board[i][i] == 'X' for i in range(3)): return 'X'
    if all(board[i][i] == 'O' for i in range(3)): return 'O'
    if all(board[i][2 - i] == 'X' for i in range(3)): return 'X'
    if all(board[i][2 - i] == 'O' for i in range(3)): return 'O'
    # Tie (no empty cell)
    if all(cell in ['X', 'O'] for row in board for cell in row):
        return 'Tie'
    return None

def ai_move(board: List[List[str]]) -> (int, int):
    # Naive random AI: pick first empty cell
    for r in range(3):
        for c in range(3):
            if board[r][c] == "":
                return r, c
    return -1, -1  # Should never happen

# --- API Endpoints ---

@app.get("/", tags=["default"])
def health_check():
    """Health check endpoint."""
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.post("/api/register", response_model=Token, tags=["auth"], summary="Register a new user")
def register_user(reg: UserRegisterRequest):
    reg_username = reg.username.strip().lower()
    reg_email = reg.email.strip().lower()
    if reg_username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    for user in users_db.values():
        if user["email"] == reg_email:
            raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(reg.password)
    user = User(
        id=str(uuid4()),
        username=reg_username,
        email=reg_email,
        hashed_password=hashed_pw
    )
    users_db[reg_username] = user.dict()
    access_token = create_access_token(data={"sub": reg_username})
    return Token(access_token=access_token, token_type="bearer")

# PUBLIC_INTERFACE
@app.post("/api/login", response_model=Token, tags=["auth"], summary="Login and get access token")
def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username.strip().lower()
    user = users_db.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username})
    return Token(access_token=access_token, token_type="bearer")

# PUBLIC_INTERFACE
@app.post("/api/game/start", response_model=Game, tags=["game"], summary="Start a new game")
def start_game(req: GameStartRequest, current_user: User = Depends(get_current_user)):
    user = current_user
    if req.opponent == "ai":
        player_x = user.username
        player_o = "AI"
    elif req.opponent == "human":
        opp_name = req.opponent_username.strip().lower() if req.opponent_username else None
        if not opp_name or opp_name == user.username:
            raise HTTPException(status_code=400, detail="Provide a valid opponent username")
        if opp_name not in users_db:
            raise HTTPException(status_code=404, detail="Opponent not found")
        player_x, player_o = sorted([user.username, opp_name])
    else:
        raise HTTPException(status_code=400, detail="Opponent type must be 'ai' or 'human'")
    # Each game has unique ID
    gid = str(uuid4())
    game = Game(
        id=gid,
        player_x=player_x,
        player_o=player_o if player_o != "AI" else "AI",
        board=[["" for _ in range(3)] for _ in range(3)],
        current_turn="X",
        status=GameStatus.ONGOING,
        winner=None
    )
    games_db[gid] = game.dict()
    moves_db[gid] = []
    return game

# PUBLIC_INTERFACE
@app.post("/api/game/move", response_model=GameStateResponse, tags=["game"], summary="Make a move in a game")
def make_move(req: MoveRequest, current_user: User = Depends(get_current_user)):
    gid = req.game_id
    user = current_user
    game_data = games_db.get(gid)
    if not game_data:
        raise HTTPException(status_code=404, detail="Game not found")
    game = Game(**game_data)
    symbol = game.current_turn
    if symbol == "X":
        player = game.player_x
    else:
        player = game.player_o
    # Check user is allowed
    if player != user.username and player != "AI":
        raise HTTPException(status_code=403, detail="It's not your turn")
    if not (0 <= req.row < 3 and 0 <= req.col < 3):
        raise HTTPException(status_code=400, detail="Row/Col must be 0-2")
    if game.board[req.row][req.col] != "":
        raise HTTPException(status_code=400, detail="Cell already taken")
    # Register move
    game.board[req.row][req.col] = symbol
    move = Move(
        player=user.username,
        row=req.row,
        col=req.col,
        symbol=symbol,
        timestamp=time.time()
    )
    moves_db[gid].append(move.dict())
    # Check for winner/tie
    outcome = check_winner(game.board)
    if outcome:
        if outcome == 'X':
            game.status = GameStatus.X_WON
            game.winner = game.player_x
            users_db[game.player_x]["games_won"] += 1
            users_db[game.player_o]["games_lost"] += 1 if game.player_o in users_db else 0
        elif outcome == 'O':
            game.status = GameStatus.O_WON
            game.winner = game.player_o
            users_db[game.player_o]["games_won"] += 1 if game.player_o in users_db else 0
            users_db[game.player_x]["games_lost"] += 1
        elif outcome == "Tie":
            game.status = GameStatus.TIE
            users_db[game.player_x]["games_tied"] += 1
            users_db[game.player_o]["games_tied"] += 1 if game.player_o in users_db else 0
        users_db[game.player_x]["games_played"] += 1
        if game.player_o in users_db:
            users_db[game.player_o]["games_played"] += 1
    else:
        # Next turn, AI move if required
        game.current_turn = "O" if symbol == "X" else "X"
        if (game.current_turn == "O" and game.player_o == "AI" and game.status == GameStatus.ONGOING):
            r, c = ai_move(game.board)
            if r >= 0:
                game.board[r][c] = "O"
                ai_mv = Move(player="AI", row=r, col=c, symbol="O", timestamp=time.time())
                moves_db[gid].append(ai_mv.dict())
                outcome = check_winner(game.board)
                if outcome:
                    if outcome == 'O':
                        game.status = GameStatus.O_WON
                        game.winner = "AI"
                        users_db[game.player_x]["games_lost"] += 1
                    elif outcome == "Tie":
                        game.status = GameStatus.TIE
                        users_db[game.player_x]["games_tied"] += 1
                    users_db[game.player_x]["games_played"] += 1
            # AI doesn't increment games_won in users_db (AI is not persistent)
        else:
            pass
    games_db[gid] = game.dict()
    response = GameStateResponse(game=game, moves=[Move(**mv) for mv in moves_db[gid]])
    return response

# PUBLIC_INTERFACE
@app.get("/api/game/state", response_model=GameStateResponse, tags=["game"], summary="Get current game state")
def get_game_state(game_id: str, current_user: User = Depends(get_current_user)):
    game_data = games_db.get(game_id)
    if not game_data:
        raise HTTPException(status_code=404, detail="Game not found")
    game = Game(**game_data)
    history = [Move(**mv) for mv in moves_db.get(game_id, [])]
    return GameStateResponse(game=game, moves=history)

# PUBLIC_INTERFACE
@app.get("/api/leaderboard", response_model=LeaderboardResponse, tags=["leaderboard"], summary="Get leaderboard standings")
def get_leaderboard():
    entries = [
        LeaderboardEntry(
            username=user["username"],
            games_won=user["games_won"],
            games_played=user["games_played"],
            games_lost=user["games_lost"],
            games_tied=user["games_tied"],
        )
        for user in users_db.values()
    ]
    # Sort by most wins, then most played, then username
    entries.sort(key=lambda e: (-e.games_won, -e.games_played, e.username))
    return LeaderboardResponse(entries=entries)

# PUBLIC_INTERFACE
@app.get("/api/game/history", response_model=List[Game], tags=["game"], summary="Get game history for current user")
def get_game_history(current_user: User = Depends(get_current_user)):
    games = [
        Game(**game)
        for game in games_db.values()
        if current_user.username in (game["player_x"], game["player_o"])
    ]
    # Sort by most recent first (assuming game id correlates with order)
    return sorted(games, key=lambda x: x.id, reverse=True)
