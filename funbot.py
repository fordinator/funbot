# Funbot - A versatile Discord bot for jokes, chess, I Ching, trivia, and music generation
# Combines multiple game features into a single Python script for easy deployment

import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional

import discord
from discord import app_commands
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

import json
import aiohttp
import re
import io
import chess
import urllib.parse

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Model and provider configuration constants
VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

# Discord embed colors for different states
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

# Message streaming configuration
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500

# Chess piece and board display emojis
PIECE_EMOJIS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟︎', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚︎',
}
SQUARE_EMOJIS = {'light': '⬜', 'dark': '⬛'}
CORNER_EMOJI = '🔳'
RANK_EMOJIS = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣']
FILE_EMOJIS = ['🇦', '🇧', '🇨', '🇩', '🇪', '🇫', '🇬', '🇭']

# Joke caching system
JOKES_CACHE_FILENAME = "jokes_cache.json"
JOKES_CACHE: list[str] = []
DIRTY_JOKES_CACHE_FILENAME = "dirty_jokes_cache.json"
DIRTY_JOKES_CACHE: list[str] = []

# Load I Ching hexagram data
with open("data/iching.json", "r", encoding="utf-8") as f:
    HEXAGRAMS = json.load(f)
    
def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Load bot configuration from YAML file."""
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)

# Global variables for game state management
turns_passed = 0

def fuzzy_match_answer(user_answer: str, correct_answer: str, threshold: float = 0.7) -> bool:
    """Check if user answer is close enough to correct answer using fuzzy matching."""
    # Normalize both answers - lowercase, strip, remove common articles/prepositions
    def normalize(text: str) -> str:
        # Remove punctuation and extra spaces, convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', text.lower().strip())
        # Remove common words that don't matter for matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in normalized.split() if word not in stop_words]
        return ' '.join(words)
    
    user_norm = normalize(user_answer)
    correct_norm = normalize(correct_answer)
    
    # Exact match after normalization
    if user_norm == correct_norm:
        return True
    
    # Check if user answer contains the correct answer or vice versa
    if user_norm in correct_norm or correct_norm in user_norm:
        return True
    
    # Simple Levenshtein-like ratio check
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, user_norm, correct_norm).ratio()
    
    return similarity >= threshold
    
def extract_image_url(resp_dict: dict) -> str:
    """
    Given a dict from anakin.ai, return just the base URL
    """
    raw = resp_dict.get("content", "")
    match = re.search(r'!\[\]\(([^)]+)\)', raw)
    if not match:
        return None
        
    full_url = match.group(1)
    clean_url = full_url.split("?")[0]
    return clean_url

# Initialize bot configuration and state
config = get_config()
curr_model = next(iter(config["models"]))
msg_nodes = {}
last_task_time = 0

# Configure Discord bot with proper intents
intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

# Game session storage - keyed by (channel_id, user_id) tuples
chess_sessions: dict[tuple[int, int], 'AsyncChessSession'] = {}
trivia_sessions = {}

# Global HTTP client for external API calls
httpx_client = httpx.AsyncClient()

def is_in_allowed_channel():
    """Decorator to restrict commands to specific channels based on config."""
    async def predicate(interaction: discord.Interaction) -> bool:
        allowed_channel_ids = config.get("permissions", {}).get("channels", {}).get("allowed_ids", {})
        if not allowed_channel_ids:
            return True
        return interaction.channel_id in allowed_channel_ids
    return app_commands.check(predicate)

@discord_bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        await interaction.response.send_message(
            "⚠️ You can only use this command in the designated channel.",
            ephemeral = True
        )
    else:
        logging.error(f"An unhandled error occurred: {error}")
        await interaction.followup.send("❌ An unexpected error occurred.", ephemeral=True)
        
# Data classes for game state management
@dataclass
class TriviaSession:
    """Manages state for a trivia game session."""
    interaction: discord.Interaction
    questions: list[dict[str, str]]
    subject: str = ""
    current_index: int = 0
    score: int = 0
    
@dataclass
class MsgNode:
    """Represents a message node in the conversation chain for AI context."""
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    

@discord_bot.tree.command(
    name="trivia",
    description="🧠 Starts a 3-question trivia game on a given subject.")
@is_in_allowed_channel()
async def trivia(interaction: discord.Interaction, subject: str):
    session_key = (interaction.channel_id, interaction.user.id)
    if session_key in trivia_sessions:
        await interaction.response.send_message("⚠️ You already have a trivia game in progress in this channel. Please finish it first!", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    system_msg = f"""
        You are a trivia-question generator.
        Search the web for simple trivia questions about {subject}.
        Return **ONLY** a valid JSON array of **exactly three** objects.
        Each object must contain "question" and "answer" keys.
        The topic for the questions is: **{subject}**.
        Generate questions based on {subject}.

        Example format (DO NOT USE THESE LITERALLY):
        [
            {{"question": "What is the capital of France?", "answer": "Paris"}},
            {{"question": "Who wrote 'Hamlet'?", "answer": "William Shakespeare"}},
            {{"question": "What is the boiling point of water at sea level in Celsius?", "answer": "100°C"}}
        ]
        """
        
    try:
        provider_config = config["providers"]["openai"]
        openai_client = AsyncOpenAI(base_url=provider_config["base_url"], api_key=provider_config.get("api_key"))
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_msg}],
            response_format={"type": "json_object"},
        )
        
        raw_json_str = response.choices[0].message.content
        logging.error(f"Complete json: {raw_json_str}")
        data = json.loads(raw_json_str)
        
        if isinstance(data, dict):
            questions_list = next((v for v in data.values() if isinstance(v, list)), None)
        else:
            questions_list = data
            
        if not questions_list or len(questions_list) < 3:
            logging.error("The AI model did not return 3 trivia questions.")
        
        session = TriviaSession(interaction=interaction, questions=questions_list, subject=subject)
        trivia_sessions[session_key] = session
        
        first_question = session.questions[0]['question']
        embed = discord.Embed(
            title=f"Trivia: {subject.title()}",
            description=first_question,
            color=EMBED_COLOR_INCOMPLETE
        )
        embed.set_footer(text=f"Question 1 of 3 | Reply to this message to answer.")
        await interaction.followup.send(embed=embed)
    
    except Exception as e:
        logging.error(f"Trivia command failed: {e}")
        await interaction.followup.send("❌ I was unable to generate trivia questions on that topic. Please try again or choose a different subject.", ephemeral=True)
        


# Chess game utilities
def create_board_embed(board: chess.Board, player_name: str, status: str) -> discord.Embed:
    """Create a Discord embed showing the current chess board state."""
    fen_string = board.fen()
    url_encoded_fen = urllib.parse.quote(fen_string)
    
    try:
        image_url = f"https://fen2image.chessvision.ai/{url_encoded_fen}"
        # Flip board perspective when it's black's turn
        if board.turn == chess.BLACK:
            image_url += "?pov=black"
    except Exception as e:
        logging.error(f"Exception creating chessboard image URL: {e}")
        image_url = ""
        
    embed = discord.Embed(
        title=f"Chess Game: {player_name} (White) vs. Funbot (Black)", 
        description=f"**{status}**",
        color=discord.Color.dark_gray()
    )
    
    if image_url:
        embed.set_image(url=image_url)
    
    turn = "White's Turn" if board.turn == chess.WHITE else "Black's Turn"
    embed.set_footer(text=f"Move {board.fullmove_number} | {turn}")
    
    return embed


class AsyncChessSession:
    """Manages a chess game session between a user and the bot."""
    ENGINE_API_URL = "https://chess-api.com/v1"
    ENGINE_DEPTH = 3
    
    def __init__(self, interaction: discord.Interaction):
        self.interaction = interaction
        self.board = chess.Board()
        self.last_eval = 0.0  # Track evaluation score for commentary
        self._session = None  # Reusable HTTP session
        
    async def speak(self, text: Optional[str], embed: Optional[discord.Embed]=None):
        """Send a message to the game channel."""
        await self.interaction.channel.send(content=text, embed=embed)
        
    async def _get_session(self):
        """Get or create HTTP session for chess engine API calls."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _query_engine(self) -> tuple[str, float, bool]:
        """Query the chess engine for the best move and evaluation."""
        payload = {"fen": self.board.fen(), "depth": self.ENGINE_DEPTH}
        session = await self._get_session()
        async with session.post(self.ENGINE_API_URL, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            move = data.get("move") or data.get("bestMove")
            if not move:
                raise RuntimeError(f"API reply missing move field: {data}")
            score = float(data.get("eval", 0))
            captured = bool(data.get("captured"))
            return move, score, captured
    
    async def cleanup(self):
        """Clean up HTTP session when game ends."""
        if self._session and not self._session.closed:
            await self._session.close()
                
    async def start_game(self):
        status = "New game started. You are white. Use `/chess <move>` to play."
        embed = create_board_embed(self.board, self.interaction.user.display_name, status)
        await self.speak(None, embed=embed)
        
    async def ai_reply(self):
        """Process AI move and provide game commentary."""
        global turns_passed 
        
        try:
            uci_move, score, captured = await self._query_engine()
            swing = score - self.last_eval  # Calculate position change
            self.last_eval = score
            
            move = chess.Move.from_uci(uci_move)
            san_move_str = self.board.san(move)
            self.board.push(move)
            
            status = f"My move: {san_move_str}."
            # Provide contextual commentary based on game state
            if captured:
                await self.speak("It appears I captured one of your pieces.")
                
            # Comment on position swings (negative swing = bad for AI)
            if swing < -.35:
                await self.speak("I apologize, but you really blew that one.")
            elif swing < -.25:
                await self.speak("Not a good idea.")
            elif swing < -.15:
                await self.speak("I'm not too sure that was the best play.") 
            
            # Positive swings (good for user)
            if swing > .35:
                await self.speak("Uh oh. I really blew that one.")
            elif swing > .25:
                await self.speak("Nice move!")
            elif swing > .15:
                await self.speak("Good show!")
                
            # Overall position commentary
            if score < -4.0:
                await self.speak("Putting the screws to you, aren't I?")
            if score > 4.0:
                await self.speak("You really got me over a barrel here.")
            if score < -10.0:
                await self.speak("I apologize, but you are going to lose.")
            if score > 10.0:
                await self.speak("Looks like I'm going to lose this one.")
            if (turns_passed % 10 == 0) and ((score > -0.5) and (score < 0.5)):
                await self.speak("Close game!")
            
            if self.board.is_checkmate():
                status += "\n**CHECKMATE!**"
                embed = create_board_embed(self.board, self.interaction.user.display_name, status)
                await self.speak(None, embed=embed)
                return True
            elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                status += "\n**It's a draw!**"
                embed = create_board_embed(self.board, self.interaction.user.display_name, status)
                await self.speak(None, embed=embed)
                return True
            elif self.board.is_check():
                status += "\n**CHECK!**"
            
            embed = create_board_embed(self.board, self.interaction.user.display_name, status)
            await self.speak(None, embed=embed)
            turns_passed += 1
            return False
            
        except Exception as e:
            logging.error(f"Chess reply failed: {e}")
            await self.speak("❌ I encountered an error trying to make my move.")
            return True
            
    async def user_move(self, san_move: str):
        try:
            move = self.board.parse_san(san_move)
            self.board.push(move)
            
            if self.board.is_game_over():
                status = "\n**CHECKMATE!!** You got me." if self.board.is_checkmate() else "**It's a draw!**"
                embed = create_board_embed(self.board, self.interaction.user.display_name, status)
                await self.speak(None, embed=embed)
                return True
                
            return await self.ai_reply()
            
        except ValueError:
            await self.speak(f"⚠️ `{san_move}` is not a valid move in standard notation.")
            return False
            
chess_group = discord.app_commands.Group(name="chess", description="Commands to play a game of chess.")

@chess_group.command(name="start", description="Starts a new chess game against Funbot.")
@is_in_allowed_channel()
async def chess_start(interaction: discord.Interaction):
    global turns_passed
    session_key = (interaction.channel_id, interaction.user.id)
    if session_key in chess_sessions:
        await interaction.response.send_message("⚠️ You already have a chess game in progress in this channel. Use `/chess resign` to end it.", ephemeral=True)
        return
        
    await interaction.response.defer()
    session = AsyncChessSession(interaction)
    chess_sessions[session_key] = session
    await session.start_game()
    turns_passed = 0
    await interaction.followup.send("**You're on!**", ephemeral=True)

    
@chess_group.command(name="move", description="Make a move in your current chess game.")
@is_in_allowed_channel()
async def chess_move(interaction: discord.Interaction, move: str):
    session_key = (interaction.channel_id, interaction.user.id)
    if session_key not in chess_sessions:
        await interaction.response.send_message("⚠️ You don't have a chess game in progress. Use `/chess start` to begin.", ephemeral=True)
        return
        
    await interaction.response.defer()
    session = chess_sessions[session_key]
    game_over = await session.user_move(move)
    
    if game_over:
        await session.cleanup()
        del chess_sessions[session_key]
        await interaction.followup.send("**GAME OVER.**", ephemeral=True)
    else:
        await interaction.followup.send(f"Move `{move}` processed.", ephemeral=True)
        
@chess_group.command(name="board", description="Shows the current board state of your game.")
@is_in_allowed_channel()
async def chess_board(interaction: discord.Interaction):
    session_key = (interaction.channel_id, interaction.user.id)
    if session_key not in chess_sessions:
        await interaction.response.send_message("⚠️ You don't have a chess game in progress. Use `/chess start` to begin.", ephemeral=True)
        return

    session = chess_sessions[session_key]
    embed = create_board_embed(session.board, interaction.user.display_name, "Current board state. Your move.")
    await interaction.response.send_message (embed=embed, ephemeral=True)
    
@chess_group.command(name="resign", description="Resignes your current chess game.")
@is_in_allowed_channel()
async def chess_resign(interaction: discord.Interaction):
    session_key = (interaction.channel_id, interaction.user.id)
    if session_key in chess_sessions:
        session = chess_sessions[session_key]
        await session.cleanup()
        del chess_sessions[session_key]
        await interaction.response.send_message("✅ You have resigned the game.", ephemeral=False)
    else:
        await interaction.response.send_message("⚠️ You don't have a chess game in progress.", ephemeral=True)

discord_bot.tree.add_command(chess_group)
        
@discord_bot.tree.command(name="compose", description="🎶 Generate a short instrumental using Suno")
@is_in_allowed_channel()
async def music(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()    
    
    url = "https://api.acedata.cloud/suno/audios"

    headers = {
        "authorization": "Bearer 721cd16b726d48458cc315905201c337"
    }

    payload = {
        "accept": "application/json",
        "content-type": "application/json",
        "action": "generate",
        "prompt": prompt,
        "model": "chirp-v3-5",
        "custom": False,
        "instrumental": True,
        "lyric": ""
    }
    
    try:
            
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                if resp.status != 200:
                    raise RuntimeError(f"Suno API error: {resp.status}: {text}")
                    
                json_payload = await resp.json()
                data_list = json_payload.get("data", [])
                
            if not data_list:
                await interaction.followup.send("⚠️ No music variants returned. Try tweaking your prompt.", ephemeral=True)
                return
            
            variants = data_list[:2]
            files = []
        
            for idx, track in enumerate(variants):      
                async with session.get(track["audio_url"]) as r:
                    r.raise_for_status()
                    audio_bytes = await r.read()
                
                files.append(discord.File(io.BytesIO(audio_bytes), f"{track['title']}.mp3"))
                
            await interaction.followup.send(
                content="🎶 **Your music files are below. Enjoy!**",
                files=files
            )
              
        
    except Exception as e:
        await interaction.followup.send("❌ Failed to generate music. Please try again later.", ephemeral=True)
        await interaction.followup.send(e)
        
        
import logging
import aiohttp
import discord

# Assume discord_bot, config, and is_in_allowed_channel are defined elsewhere

@discord_bot.tree.command(
    name="imagine",
    description="🎨 Generate an image via Stable Diffusion"
)
@is_in_allowed_channel()
async def imagine(
    interaction: discord.Interaction,
    prompt: str,
    guidance_scale: float = 3.5,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = -1,
    checker: bool = True
):
    await interaction.response.defer()
    
    await interaction.edit_original_response(content="Generating image...")
    
    anakin_api_key = config["providers"]["anakin"]["api_key"]
    app_id = "32271" 
    url = f"https://api.anakin.ai/v1/quickapps/{app_id}/runs"

    headers = {
        "Authorization": f"Bearer {anakin_api_key}",
        "X-Anakin-Api-Version": "2024-05-06",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "checker": True
        },
        "stream": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()

                data = await resp.json(content_type=None)
                image_markdown = data.get("content")

                if image_markdown and image_markdown.startswith("![](http"):
                    # Strip the fucking markdown bullshit off the URL
                    clean_url = image_markdown.strip("![]()")
                    
                    # Send the clean URL. Discord will embed it.
                    await interaction.edit_original_response(content=clean_url)
                else:
                    await interaction.edit_original_response(content=f"❌ **Error:** API response was missing or malformed. Response: `{data}`")

    except Exception as e:
        logging.exception("Error in /imagine")
        await interaction.followup.send(f"❌ Fucking unexpected error: {e}", ephemeral=True)

@discord_bot.tree.command(name="iching", description="☯️ Ask the I Ching a question and receive a hexagram")
@is_in_allowed_channel()
async def iching(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    
    url = (
        "https://www.random.org/integers/"
        "?num=6&min=6&max=9&col=1&base=10&format=plain&rnd=new"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                text = await resp.text()
    except Exception as e:
        return await interaction.followup.send(
            "⚠️ Could not fetch true randomness. Please try again shortly."
        )

    nums = [int(n) for n in text.split() if n.strip()]
    key = "".join("1" if n in (7, 9) else "0" for n in nums)

    entry = HEXAGRAMS.get(key)
    if not entry:
        return await interaction.followup.send(
            f"❓ I-Ching cast failed (invalid key `{key}`)."
        )

    embed = discord.Embed(
        title=entry["name"],
        url=entry["url"],
        description=f"🔮 **Your question:** {question}",
        color=0x8B4513
    )
    embed.add_field(name="Numbers cast", value=", ".join(map(str, nums)))
    embed.set_footer(text="I Ching via Random.org")

    await interaction.followup.send(embed=embed)


async def fetch_jokes() -> None:
    cache_filename = JOKES_CACHE_FILENAME
    cache = JOKES_CACHE
    
    
    if cache:
        return
        
    try:
        with open(cache_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all (isinstance(item, str) for item in data):
            cache.extend(item.strip() for item in data if item.strip())
    except Exception as e:
        logging.error(f"Exception loading jokes: {e}")
        
        
async def tell_random_joke() -> str:
    await fetch_jokes()
    cache = JOKES_CACHE
    
    if not cache:
        return "I'm all out of jokes now. Sorry, bub."
    
    try:
        async with aiohttp.ClientSession() as session:
            max_idx = len(cache)
            url = f"https://www.random.org/integers/?num=1&min=1&max={max_idx}&col=1&base=10&format=plain&rnd=new"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Referer": "https://random.org/integers/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*,q=0.8"
            }
            async with session.get(url, headers=headers, timeout=10) as resp:
                resp.raise_for_status()
                text = await resp.text()
                idx = int(re.findall(r"\d+", text)[0]) - 1
    except Exception as e:
        logging.error(f"Error getting random number: {e}")
        
    return cache.pop(idx)
    
    
@discord_bot.tree.command(name="joke", description="😂 Tell a random joke")
async def joke(interaction: discord.Interaction):
    await interaction.response.defer()
    joke_text = await tell_random_joke()
    
    embed = discord.Embed(
        description=joke_text,
        color=EMBED_COLOR_COMPLETE
    )
    
    await interaction.followup.send(embed=embed)
        

@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time
    
    # Handle trivia game responses
    session_key = (new_msg.channel.id, new_msg.author.id)
    if session_key in trivia_sessions:
        session = trivia_sessions[session_key]
        
        user_answer = new_msg.content.strip()
        correct_answer = session.questions[session.current_index]['answer']
        
        # Check if the answer is correct using fuzzy matching
        is_correct = fuzzy_match_answer(user_answer, correct_answer)
        if is_correct:
            session.score += 1
            feedback = "✅ **Correct!**"
            color = EMBED_COLOR_COMPLETE
        else:
            feedback = "❌ **Incorrect.**"
            color = discord.Color.red()
        
        answer_embed = discord.Embed(
            title=f"Question {session.current_index + 1} - {feedback}",
            description=f"**Correct Answer:** {correct_answer}\n**Your Answer:** {user_answer}",
            color=color
        )
        await new_msg.channel.send(embed=answer_embed)
        
        session.current_index += 1
        if session.current_index >= len(session.questions):
            # Determine final score message
            final_score = session.score
            total_questions = len(session.questions)
            percentage = (final_score / total_questions) * 100
            
            if percentage == 100:
                score_message = "🏆 **Perfect score!** You got them all!"
            elif percentage >= 66:
                score_message = "🎉 **Well done!** Great job!"
            elif percentage >= 33:
                score_message = "👍 **Not bad!** Keep it up!"
            else:
                score_message = "📚 **Better luck next time!** Practice makes perfect!"
            
            end_embed = discord.Embed(
                title="🧠 Trivia Complete!",
                description=f"**Final Score:** {final_score}/{total_questions} ({percentage:.0f}%)\n\n{score_message}",
                color=EMBED_COLOR_COMPLETE if percentage >= 66 else EMBED_COLOR_INCOMPLETE
            )
            await new_msg.channel.send(embed=end_embed)
            del trivia_sessions[session_key]
        else:
            next_question = session.questions[session.current_index]['question']
            question_embed = discord.Embed(
                title=f"Trivia: {session.subject.title()}",
                description=next_question,
                color=EMBED_COLOR_INCOMPLETE
            )
            question_embed.set_footer(text=f"Question {session.current_index + 1} of 3 | Score: {session.score}/{session.current_index} | Reply to answer")
            await new_msg.channel.send(embed=question_embed)
            
        return
        
    # Main AI chat handling (only process if bot is mentioned or in DMs)
    is_dm = new_msg.channel.type == discord.ChannelType.private
    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or config["allow_dms"] if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.split("/", 1)
    model_parameters = config["models"].get(provider_slash_model, None)

    base_url = config["providers"][provider]["base_url"]
    api_key = config["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config["max_text"]
    max_images = config["max_images"] if accept_images else 0
    max_messages = config["max_messages"]

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config["system_prompt"]:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
    
    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))
    
    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()
        
    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                finish_reason = curr_chunk.choices[0].finish_reason

                prev_content = curr_content or ""
                curr_content = curr_chunk.choices[0].delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time
                    
                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:

                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


# Bot startup
async def main() -> None:
    """Start the Discord bot with configured token."""
    await discord_bot.start(config["bot_token"])

if __name__ == "__main__":
    asyncio.run(main())
