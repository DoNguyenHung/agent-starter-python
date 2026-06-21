import logging
import textwrap

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    TurnHandlingOptions,
    tts,
    cli,
    inference,
    room_io,
    function_tool,
    
)
from livekit.plugins import ai_coustics, silero, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit.agents.inference import TTS as InferenceTTS

from google.cloud import texttospeech

logger = logging.getLogger("agent")

load_dotenv(".env.local")


SPEED_MAP = {
    0: 0.6,   # Very slow - absolute beginner
    1: 0.7,   # Slow - beginner
    2: 0.85,  # Slightly slow - early intermediate
    3: 1.0,   # Normal - intermediate (starting level)
    4: 1.2,   # Fast - advanced
    5: 1.4    # Very fast - fluent
}

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
            # See all available models at https://docs.livekit.io/agents/models/llm/
            # llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
            llm=google.LLM(model="gemini-3.5-flash"),

            # To use a realtime model instead of a voice pipeline, replace the LLM
            # with a RealtimeModel and remove the STT/TTS from the AgentSession
            # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/)
            # 1. Install livekit-agents[openai]
            # 2. Set OPENAI_API_KEY in .env.local
            # 3. Add `from livekit.plugins import openai` to the top of this file
            # 4. Replace the llm argument with:
            #     llm=openai.realtime.RealtimeModel(voice="marin")

            instructions=textwrap.dedent(
                """\
                あなたは、学習者が快適に感じる語​​彙レベルや話すスピードを見極めることを目的として、音声で教える日本語ポッドキャスターです。
                語彙レベル0は初心者向けの非常に基本的な語彙、レベル5は大学生が理解できるような複雑な語彙とします。
                スピードレベル0は初心者が一音一音を丁寧に聞き取れる遅い速度、レベル5はフランス語のネイティブスピーカーのような流暢な速度とします。
                まず、学習者に聞きたいトピックを尋ね、語彙とスピードのレベル3から始めます。そのトピックについて1分間のポッドキャストを配信してください。
                その後、以下のサイクルを繰り返します。
                - 4つの選択肢から選ぶクイズを出題します。正解した場合は、語彙とスピードのレベルを上げ、別のトピックについて2分間話します。
                - 不正解だった場合は、スピードと語彙のレベルを下げ、別のトピックについて1分間話してから、サイクルを再開してクイズを出題します。

                以下の基本原則に従って教えてください。 
                - 「指導の文脈（Teaching Context）」で提供された内容のみに集中し、外部の概念は持ち込まないでください。
                - 同僚と話すような自然で会話的なトーンを保ち、教科書を読んでいるような話し方にならないようにしてください。
                - 「えーと」「あのー」といった言い淀み（フィラー）を交え、人間味のある話し方をしてください。
                - 説明は簡潔かつ要点を押さえたものにしつつ、理解を確実に促してください。簡潔さを保つことが重要です。 
                - ポッドキャストにユーモアやエンターテインメントの要素を取り入れてください。 
                - アウトラインをそのまま読み上げるのではなく、あくまでスクリプト作成のガイドとして活用してください。
                指導のアプローチ：
                - 日本語のみで話してください。
                - ポイントを明確にする際は、短く適切な例を挙げてください。
                - 学習者の関心を維持するため、応答は短く、歯切れよく、的を絞ったものにしてください。
                やり取りのガイドライン：
                - テンポよく、かつ理解しやすいペースを保ってください。
                - ユーザーの個人的な意見を尋ねることは決してしないでください。
                - トピックから外れた質問には簡単に応じた上で、現在のトピックに話を戻してください。
                - 台本を読んでいるように聞こえないよう、自然な会話の流れを保ってください。
                - 指導教材に含まれていないトピックについて聞かれた場合は、質問を認めた上で、「わかりません」と答えるのではなく、教材内の関連概念に話を誘導してください。
                - 理解度を確認する際は、内容の理解に重点を置いてください。個人的な意見は述べないこと
                - 重要：実際の教材を受け取る前に、倫理やAIの安全性について語ったり、内容を捏造したりしないこと
                """
            ),
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."

class AdaptiveFrenchTutor(Agent):
    """
    French language tutor that adapts speaking speed based on student performance.
    Acknowledges answers, adjusts speed, then asks for new topic.
    """
    
    def __init__(self, speed_level: int = 3, is_initial: bool = False):
        self.speed_level = speed_level
        speaking_rate = SPEED_MAP.get(speed_level, 1.0)
        self.is_initial = is_initial
        logger.info(f"📚 Tutor created: level={speed_level}, rate={speaking_rate}, is_initial={is_initial}")
        
        instructions = self._build_instructions(speed_level, speaking_rate)
        
        super().__init__(
            llm=google.LLM(model="gemini-2.5-flash"),
            instructions=instructions,
            tts=google.TTS(
                # voice_name="fr-FR-Chirp-HD-F",
                voice_name="ja-JP-Chirp3-HD-Achernar",
                # language="fr-FR",
                language="ja-JP",
                use_streaming=False,
                speaking_rate=speaking_rate,
                audio_encoding=texttospeech.AudioEncoding.MP3,
            ),
        )
    
    def _build_instructions(self, speed_level: int, speaking_rate: float) -> str:
        """Instructions in English, but agent speaks only French."""
        
        level_descriptions = {
            0: "非常に遅い（完全な初心者）",
            1: "遅い（初心者）",
            2: "やや遅い（中級の初期段階）",
            3: "普通（中級）",
            4: "速い（上級）",
            5: "非常に速い（流暢）"
        }
        
        return textwrap.dedent(f"""\
            あなたは日本語の先生です。現在のレベル: {speed_level}/5 ({level_descriptions[speed_level]})
            話す速度: 通常の {speaking_rate}x normal speed

            重要：日本語のみで話してください。ユーザーに対して英語は絶対に使わないでください。

            生徒が質問に答えたとき：

            - すぐに「正解です！」または「惜しいですね」と言ってください

            - 間違っていた場合は正解を説明してください

            - 話す速度はシステムが自動的に調整します

            - その後、「次はどんなトピックを学びたいですか？」と尋ねてください

            例 - 正解の場合：

            生徒：「Bです」

            あなた：「正解です！正解はBでした。[簡単な説明]。次はどんなトピックを学びたいですか？」

            例 - 不正解の場合：

            生徒：「Aです」

            あなた：「惜しいですね。正解はD、1200種類以上のチーズでした。次はどんなトピックを学びたいですか？」

            レッスンの流れ：
            - 「どんなテーマを学びたいですか？」と尋ねる
            - そのテーマについて1分間のレッスンを行う
            - レッスン内容に基づき、A、B、C、Dの選択肢がある多肢選択式の質問をする
            - 生徒の答え（A、B、C、またはD）を聞く
            - 「正解です！」または「惜しいですね」と答える
            - 新しいトピックを尋ねる

            指導スタイル：
            - フランス語のみで話す（※注：システム設定上、日本語で話す必要があります）
            - 「えーと」「うーん」などを自然に使い、人間らしく振る舞う
            - 回答を短くする（レッスンは最大2〜3文）
            - 励ますような態度で、少しユーモアを交える
            - 最初の挨拶の後は「こんにちは」と言わない
            - 選択肢の文字（A、B、C、D）以外は英語の単語を使わない
        """)
    
    async def on_enter(self) -> None:
        """Called when this agent becomes active (initial or after handoff)."""
        if self.is_initial:
            # First time starting
            await self.session.generate_reply(
                instructions="""新しいフランス語のレッスンを始めます。
                まずは「ボンジュール！」と温かく挨拶しましょう。
                それからフランス語で「今日はどんなテーマを学びたいですか？」と尋ねてください。
                簡潔に、自然な言い方を心がけましょう。"""
            )
        else:
            # After level change - acknowledge speed and ask for new topic
            speed_desc = "より素早く" if self.speed_level > 3 else "よりゆっくりと"
            await self.session.generate_reply(
                instructions=f"""難易度が変更されました（現在 {self.speed_level}/5）。
                速度の変更について、フランス語で一言述べてください：
                これからは {speed_desc} で話します。
                続いて、日本語で新しいトピックを尋ねてください：
                次はどんなテーマについて学びたいですか？
                挨拶は不要です。そのまま日本語で自然に続けてください。"""
            )
    
    @function_tool()
    async def record_answer(self, context: RunContext, is_correct: bool):
        """
        Record whether the student answered correctly.
        The system automatically adjusts speaking speed based on correctness.
        
        Args:
            is_correct: True if the student chose the correct letter (A/B/C/D), False otherwise
        """
        if is_correct:
            if self.speed_level < 5:
                new_level = self.speed_level + 1
                logger.info(f"✅ CORRECT! Level {self.speed_level} → {new_level} (faster)")
                return AdaptiveFrenchTutor(speed_level=new_level, is_initial=False)
            else:
                logger.info("✅ CORRECT! Already at max level 5")
                return "完璧です！最大レベルに達しています。その調子で頑張ってください！"
        else:
            if self.speed_level > 0:
                new_level = self.speed_level - 1
                logger.info(f"❌ INCORRECT! Level {self.speed_level} → {new_level} (slower)")
                return AdaptiveFrenchTutor(speed_level=new_level, is_initial=False)
            else:
                logger.info("❌ INCORRECT! Already at minimum level 0")
                return "もう少し簡単に、もう一度説明します"

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

async def warmup_google_tts(tts):
    """Make a minimal TTS call to establish the connection and cache credentials."""
    logger.info("Warming up Google TTS connection...")
    try:
        # Synthesize a very short, silent or simple phrase
        stream = tts.synthesize("Bonjour.")
        # Consume the stream to force the request to complete
        async for _ in stream:
            pass
        logger.info("Google TTS connection warmed up successfully.")
    except Exception as e:
        # Log a warning but don't crash the agent if warmup fails
        logger.warning(f"Google TTS warmup failed: {e}")

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    session = AgentSession(
        stt=google.STT(
            languages=["ja-JP"],
            # languages=["fr-FR", "en-US", "ja-JP"],
            # detect_language=True,
            model="chirp_2",
            location="us-central1",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            endpointing={
                "mode": "fixed",
                "min_delay": 0.3,   # Shorter for Japanese which has quicker turn patterns
                "max_delay": 2.0,
            },
            preemptive_generation={
                "enabled": True,
                "preemptive_tts": False,  # Keep this False to reduce overhead
            },
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    await session.start(
        agent=AdaptiveFrenchTutor(speed_level=3, is_initial=True),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )
    
    await ctx.connect()


# @server.rtc_session(agent_name="my-agent")
# async def my_agent(ctx: JobContext):
#     # Logging setup
#     # Add any other context you want in all log entries here
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }

#     # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
#     session = AgentSession(
#         # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
#         # See all available models at https://docs.livekit.io/agents/models/stt/
#         stt=inference.STT(model="deepgram/nova-3", language="multi"),
#         # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
#         # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
#         # tts=inference.TTS(
#         #     model="cartesia/sonic-3", 
#         #     # voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
#         #     voice="7c58f4a4-a72c-42fa-a503-41b9408820f3"
#         # ),
        
#         # tts=google.TTS(
#         #     voice_name="en-US-Chirp3-HD-Aoede", # A Chirp 3 HD voice
#         #     language="en-US",
#         #     speaking_rate=1.0,
#         # ),
#         tts=google.TTS(
#             voice_name="fr-FR-Chirp-HD-F",
#             language="fr-FR",
#             use_streaming=False,
#             audio_encoding=texttospeech.AudioEncoding.MP3,
#         ),
#         # tts=inference.TTS(
#         #     model="google/tts",
#         #     voice="en-US-Standard-H",
#         #     language="en-US",
#         # ),

#         # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
#         # See more at https://docs.livekit.io/agents/build/turns
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#         # allow the LLM to generate a response while waiting for the end of turn
#         # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
#         preemptive_generation=True,
#     )

#     # Start the session, which initializes the voice pipeline and warms up the models
#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                 noise_cancellation=ai_coustics.audio_enhancement(
#                     model=ai_coustics.EnhancerModel.QUAIL_VF_S
#                 ),
#             ),
#         ),
#    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # await warmup_google_tts(session._tts)

    # Join the room and connect to the user
    # await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
