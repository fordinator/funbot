<h1 align="center">
  funbot
</h1>

<h3 align="center"><i>
  Play fun games on Discord!
</i></h3>

funbot provides an easy, engaging way to play simple LLM-powered games via _Discord_ channels. It works with practically any LLM, remote or locally hosted. Based on [llmcord](https://github.com/jakobdylanc/llmcord) by @jakobdylanc.

## Features

### Play chess against a genius:
Full-fledged, grand master level chess engine via `pychess.py`.

- Play chess via slash commands with a bot that responds verbally to your moves
- Display images per turn with an updated board state
- Customize the bot with ELO ratings from 1600 to 2800

### Play simple three question trivia games about any topic:
- Give the bot a topic for trivia and the LLM will search the web and generate three questions
- Answer and the bot will score you based on "fuzzy logic"
- Near infinite replayability

### Compose songs using the AceData API for Suno AI music generation
- Create a simple two minute instrumental song in any style
- Generates two files as embeds you can download
- Modifying the code can yield further options

### Scry into the future
- Generate random _I Ching_ readings using the coin method with numbers generated from `RANDOM.ORG`
- Matched with a JSON file of hexagrams, displaying the name, number and any changing lines
- Web link to a modern, dynamic interpretation of the hexagram

### Pun based Dad joke engine
- Tells simple punny Dad jokes out of an index file
- Comes with 200 jokes curated from the web to start
- Index file can be fully customized for your own jokes

## CUSTOM FUNBOT COMMANDS

`/chess start`: Start a new chess game against the 'bot. The player is always White. The 'bot is always Black.

`/chess move <move>`: Move a chess piece. Use standard chess notation -- for example, `e2e4`.

`/chess board`: Show the current board state.

`/chess resign`: Give up in the face of funbot's overwhelming genius.

`/trivia <topic>`: Start a three question trivia game about the topic specified. The LLM searches the web for your topic and generates the questions. Some questions may be obscure or formatted incorrectly, the 'bot will return an error if it doesn't work. The 'bot will give you the correct answer after each guess and attempt to score it with "fuzzy logic" (to allow for misspellings and minor words). (NOTE: This is the only part of the bot that actually makes use of the LLM. It requires access to OpenAI's `gpt-4o-search-preview` model to function.)

`/compose <song>`: Compose a 2 - 3 minute instrumental song in the style specified. Not working for now.

`/iching <question>`: Ask the 'bot the question in the prompt and receive an I-Ching reading as determined by `RANDOM.ORG.` The Hexagram, changing lines, and interpretation are displayed.

`/joke`: Tell what basically amounts to a Dad Joke. The Dad Jokes can be customized in `jokes_cache.json`. There are about 200 to start, they are removed from the cache when told, so the bot never repeats jokes.

## Instructions (from llmcord repo)

1. Clone the repo:
   ```bash
   git clone https://github.com/fordinator/funbot
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile.<br /><br />**Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments. (Default: `100,000`) |
| **max_images** | The maximum number of image attachments allowed in a single message. (Default: `5`)<br /><br />**Only applicable when using a vision model.** |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped. (Default: `25`) |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often. (Default: `false`)<br /><br />**Also disables streamed responses and warning messages.** |
| **allow_dms** | Set to `false` to disable direct message access. (Default: `true`) |
| **permissions** | Configure access permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`.<br /><br />Control which `users` are admins with `admin_ids`. Admins can change the model with `/model` and DM the bot even if `allow_dms` is `false`.<br /><br />**Leave `allowed_ids` empty to allow ALL in that category.**<br /><br />**Role and channel permissions do not affect DMs.**<br /><br />**You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control channel permissions in groups.** |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and optional `api_key` entry. Popular providers (`openai`, `ollama`, etc.) are already included.<br /><br />**Only supports OpenAI compatible APIs.**<br /><br />**Some providers may need `extra_headers` / `extra_query` / `extra_body` entries for extra HTTP data. See the included `azure-openai` provider for an example.** |
| **models** | Add the models you want to use in `<provider>/<model>: <parameters>` format (examples are included). When you run `/model` these models will show up as autocomplete suggestions.<br /><br />**Refer to each provider's documentation for supported parameters.**<br /><br />**The first model in your `models` list will be the default model at startup.**<br /><br />**Some vision models may need `:vision` added to the end of their name to enable image support.** |
| **system_prompt** | Write anything you want to customize the bot's behavior!<br /><br />**Leave blank for no system prompt.**<br /><br />**You can use the `{date}` and `{time}` tags in your system prompt to insert the current date and time, based on your host computer's time zone.** |

### Licensing and Usage

Steal it, take credit for it, change it, improve it, sell it, it's yours.

If you like this stupid thing, pop on up to the upper right there and hit a "Star" on this repo so it'll move up on Google and others can find it.

If you want to chat with me, visit https://discord.com/invite/gAugxKBHQY.

If you want to encourage other coding projects like this, donate to https://www.patreon.com/vexation1977


