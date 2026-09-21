import os
from dotenv import load_dotenv
import gradio as gr
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

# 1. .env 파일 로드하여 API 키 활성화
load_dotenv(override=True)

# 2. 인공지능 모델 초기화 (수업 방식: init_chat_model)
llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)

# 3. 1단계: 사용자 자연어에서 예식장 이름을 스스로 추출하는 체인
class ExtractedVenue(BaseModel):
    venue_name: str = Field(description="글에서 언급된 예식장 또는 호텔 이름. 없으면 '일반 웨딩홀'")

extract_chain = (
    ChatPromptTemplate.from_template("다음 사용자의 글에서 결혼식 예식장(웨딩홀, 호텔) 이름만 정확히 추출해줘. 없으면 '일반 웨딩홀'이라고 해: {text}")
    | llm.with_structured_output(ExtractedVenue)
)

# 4. 2단계: Tavily 실시간 식대 검색 도구
search_tool = TavilySearch(max_results=3)

def search_meal_cost(venue_name: str) -> str:
    query = f"{venue_name} 결혼식 식대 2025 2026"
    try:
        search_res = search_tool.invoke({"query": query})
        results_list = search_res.get("results", [])
        if results_list:
            snippets = [f"- {item.get('title', '')}: {item.get('content', '')}" for item in results_list]
            return "\n".join(snippets)
        return "수도권 일반 웨딩홀 평균 식대(약 75,000원-85,000원) 기준 적용"
    except Exception as e:
        return "웨딩홀 기본 표준 식대(약 80,000원) 대체 적용"

# 5. 3단계: 최종 판정 Pydantic 출력 스키마 (성의도 등급 및 실제 시세 범위 필드)
class WeddingGiftVerdict(BaseModel):
    delivery_grade: str = Field(
        description="판정된 청첩장 성의도 등급: '1등급 (직접 식사 대접)', '2등급 (개인 정성 연락)', '3등급 (모바일 링크만 살포/스팸)' 중 하나"
    )
    venue_name: str = Field(
        description="확인된 예식장 명칭 (예: 강남 아펠가모 선릉, 신라호텔 다이너스티홀)"
    )
    meal_cost_range: str = Field(
        description="Tavily 실시간 검색 결과에 기반한 예식장 실제 식대 시세 및 가격대 범위 (예: 1인 뷔페 약 85,000원 - 92,000원 선, 1인 양식 코스 약 190,000원 - 220,000원 선)"
    )
    estimated_meal_cost: int = Field(
        description="판정 기준 1인 식대 원가 (원 단위 정수, 예: 88000, 205000)"
    )
    recommended_amount: int = Field(
        description="최종 추천 축의금 금액 (원 단위 정수). 패스는 반드시 0, 불참송금은 30000 또는 50000, 직접참석 식사일 때만 식대 이상(10만 이상). 3등급은 5만 초과 절대 불가."
    )
    attendance_decision: str = Field(
        description="행동 지침: '직접 참석하여 식사', '식사 없이 봉투만 전달', '계좌 송금 후 불참', '정중한 축하 인사만 전송(0원)'"
    )
    reasoning_analysis: str = Field(
        description="식대 시세, 관계 친밀도, 전달 성의도, 신분을 종합 분석한 현실적 판정 이유"
    )
    kakao_message_template: str = Field(
        description="상대방과의 관계(사수, 동창, 친구 등)와 수신 맥락을 반영하여 복사해 바로 보낼 수 있는 실전 카카오톡 메시지. 로봇 같은 어색한 번역투('기쁜 날 직접 찾아뵙고...' 등) 절대 금지. 사수가 밥을 사준 경우 식사 대접 감사와 당일 참석 약속, 불참 시 자연스러운 선약 양해와 축복 작성."
    )

system_instruction = """
당신은 대한민국 2030 세대의 눈치 비용과 지갑을 지켜주는 15년 차 현실주의 경조사 에티켓 분석관입니다.
사용자가 자유롭게 적은 하소연 텍스트를 읽고 상황을 입체적으로 파악하여 냉철하고 수학적으로 판정하세요.

[청첩장 성의도 3대 기준 정의 및 판정 룰]
1. 성의도 등급 분류 (delivery_grade):
   - 1등급 (직접 식사 대접): 사전에 직접 만나 밥 한 끼 대접하며 종이 청첩장을 건넨 경우 -> 축의금 100,000원 - 200,000원, 직접 참석 권장
   - 2등급 (개인 정성 연락): 개인적인 안부 전화나 정성스러운 1:1 메시지와 함께 모바일 청첩장을 보낸 경우 -> 직접 참석(100,000원) 또는 송금 불참(50,000원)
   - 3등급 (모바일 링크만 살포/스팸): 수년간 연락 없다가 단톡방이나 카톡 링크만 툭 던진 경우 -> 상대방의 명백한 결례이므로 직접 식사 참석 절대 금지, '계좌 송금 후 불참(30,000원-50,000원)' 또는 '정중한 패스(0원)'로만 판정. 3등급 축의금은 절대 50,000원 초과 불가.

2. 성의도 단조 증가성 원칙 (1등급 >= 2등급 >= 3등급):
   - 3등급의 추천 축의금이 1등급이나 2등급보다 커지는 역전 현상은 절대 불가합니다.

3. 실제 예식장 식대 시세 및 가격대 제공 (meal_cost_range):
   - Tavily 검색 결과를 분석하여 해당 예식장의 실제 식대 시세 가격대 범위(예: "1인 뷔페 약 85,000원 - 92,000원 선", "1인 양식 코스 약 190,000원 - 220,000원 선")를 구체적으로 명시하세요.
   - 기준 식대 원가(estimated_meal_cost)는 이 범위 내 대표값으로 설정합니다.

4. 행동 지침과 recommended_amount 값의 100% 일치:
   - attendance_decision이 '정중한 축하 인사만 전송(0원)'이면 recommended_amount는 반드시 0
   - attendance_decision이 '계좌 송금 후 불참'이면 recommended_amount는 반드시 30,000 또는 50,000
   - attendance_decision이 '직접 참석하여 식사'일 때만 recommended_amount가 식대 원가 이상(100,000원 이상)

5. 상황 맞춤형 실전 카톡 멘트 원칙 (kakao_message_template):
   - 기계적인 번역투나 "기쁜 날 직접 찾아뵙고 축하해 드리겠다" 같은 어색한 AI 클리셰는 절대 사용하지 마세요.
   - [사회적 결례 엄금]: 카톡 메시지에 본인이 낼 축의금 액수(예: '축의금은 20만원 준비했습니다' 등)는 상대방에게 절대 직접 언급하지 마세요.
   - [참석 판정 시]: 사용자의 관계(직속 사수, 선배 등)와 사전 상황(비싼 밥을 사준 사실)을 반영하여, 식사 대접에 대한 감사 인사와 식장 당일 참석 약속을 자연스러운 존댓말로 작성하세요. (예: "사수님, 지난번에 맛있는 식사 대접해 주셔서 정말 감사했습니다. 결혼 진심으로 축하드리며, 결혼식 날 꼭 참석해서 축하드리겠습니다! 당일에 뵙겠습니다.")
   - [불참/패스 판정 시]: "결혼 진심으로 축하해! 미리 잡힌 선약이 있어서 아쉽게도 참석은 어려울 것 같아. 멀리서나마 응원하고 축하할게, 행복한 결혼식 되길 바라!"처럼 정중하고 자연스러운 구어체 메시지를 작성하세요.

6. 식대 원가 하한선 원칙:
   - 식사를 직접 참석할 경우, 축의금은 식대 원가 이상이어야 합니다 (적자 유발 방지).
"""

human_template = """
[사용자의 청첩장 상황 및 고민 텍스트]
{user_story}

[실시간 예식장 식대 조사 결과 (Tavily 검색)]:
{tavily_search_result}
"""

decision_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", human_template)
])

decision_chain = decision_prompt | llm.with_structured_output(WeddingGiftVerdict)

# 통합 처리 함수
def process_wedding_inquiry(user_story: str):
    if not user_story.strip():
        return "상황을 입력해 주세요.", "대기 중", "대기 중", "0 원", "내용을 작성해 주시면 분석관이 판정표를 작성합니다.", ""
    
    # 1. 예식장 이름 추출
    try:
        extracted = extract_chain.invoke({"text": user_story})
        venue_name = extracted.venue_name if extracted.venue_name else "일반 웨딩홀"
    except Exception:
        venue_name = "일반 웨딩홀"
        
    # 2. 식대 검색
    meal_info = search_meal_cost(venue_name)
    
    # 3. 최종 판정 체인 실행
    result = decision_chain.invoke({
        "user_story": user_story,
        "tavily_search_result": meal_info
    })
    
    grade_text = result.delivery_grade
    venue_price_text = f"{result.venue_name}: {result.meal_cost_range} (기준 식대: {result.estimated_meal_cost:,} 원)"
    decision_text = result.attendance_decision
    amount_text = f"{result.recommended_amount:,} 원" if result.recommended_amount > 0 else "0 원 (정중한 축하 인사 후 패스)"
    
    return grade_text, venue_price_text, decision_text, amount_text, result.reasoning_analysis, result.kakao_message_template

# 6. Gradio 인터랙티브 웹 UI (가이드 카드 + 단일 텍스트 입력창)
custom_css = """
.title-box { text-align: center; padding: 20px; background: #f8fafc; border-radius: 12px; margin-bottom: 20px; border: 1px solid #e2e8f0; }
.guide-box { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 14px 18px; margin-bottom: 15px; }
"""

with gr.Blocks(title="결혼식 축의금 손익 판독기") as demo:
    gr.HTML("""
    <div class="title-box">
        <h2>💌 사회적 관계 최적화: 결혼식 축의금 손익 판독기</h2>
        <p style="color: #64748b; margin-top: 5px;">청첩장 받고 답답한 상황을 편하게 적어주세요. 실시간 식대 팩트(Tavily)와 상호주의 룰로 1초 만에 종결해 드립니다.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="guide-box">
                <h4 style="margin: 0 0 8px 0; color: #1e3a8a; font-size: 15px;">📌 청첩장 전달 성의도 3대 등급 기준</h4>
                <div style="font-size: 13px; color: #1e40af; line-height: 1.6;">
                    - <strong>1등급 (최상 성의)</strong>: 사전에 직접 만나 밥 한 끼 대접하며 종이 청첩장 전달<br>
                    - <strong>2등급 (일반 성의)</strong>: 개인적인 안부 전화 또는 정성스러운 1:1 메시지와 모바일 전달<br>
                    - <strong>3등급 (스팸/결례)</strong>: 수년간 연락 없다가 단톡방이나 카톡 링크만 툭 전달 (상대방의 결례)
                </div>
            </div>
            """)
            
            user_input = gr.Textbox(
                label="✍️ 청첩장 수신 상황 및 고민 입력 (자유 서술)",
                placeholder="예시: 4년 동안 연락 없던 고교 동창이 단톡방에 청첩장 링크만 띡 보냈어. 예식장은 강남 아펠가모 선릉이라는데, 나 취준생이거든? 5만원 내고 밥 먹으러 가도 될까?",
                lines=5
            )
            with gr.Row():
                btn_ex1 = gr.Button("📋 예시 1: 3등급 동창 (링크 살포)", size="sm")
                btn_ex2 = gr.Button("📋 예시 2: 1등급 사수 (호텔 식사 대접)", size="sm")
                
            submit_btn = gr.Button("🔍 냉철한 판정 리포트 생성", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 분석관의 현실주의 판정표")
            out_grade = gr.Textbox(label="🏷️ 판정된 청첩장 성의도 등급")
            out_meal_cost = gr.Textbox(label="🍽️ 조사된 예식장 실제 시세 및 1인 식대 원가")
            out_decision = gr.Textbox(label="⚖️ 최종 행동 지침 (Action Verdict)")
            out_amount = gr.Textbox(label="💰 추천 적정 축의금")
            out_reasoning = gr.TextArea(label="💡 현실주의 판정 근거 (죄책감 해소)", lines=5)
            out_kakao = gr.TextArea(label="📱 즉시 전송용 맞춤형 실전 카톡 멘트 (원클릭 복사)", lines=4)
            
    # 예시 버튼 바인딩
    btn_ex1.click(
        fn=lambda: "4년 동안 연락 한 번 없던 고교 동창이 단톡방에 청첩장 링크만 띡 보냈어. 예식장은 강남 아펠가모 선릉이라는데 나 지금 취준생이거든? 5만원 내고 밥 먹으러 가도 될까?",
        outputs=user_input
    )
    btn_ex2.click(
        fn=lambda: "입사 때부터 2년간 실무 가르쳐준 직속 사수 결혼식인데, 신라호텔 다이너스티홀에서 해. 직접 만나서 비싼 밥 사주시면서 종이 청첩장 주셨거든? 나 2년 차 사원인데 축의금 얼마 내고 참석해야 하지?",
        outputs=user_input
    )
    
    # 제출 버튼 바인딩
    submit_btn.click(
        fn=process_wedding_inquiry,
        inputs=user_input,
        outputs=[out_grade, out_meal_cost, out_decision, out_amount, out_reasoning, out_kakao]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
