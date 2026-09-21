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
search_tool = TavilySearch(max_results=2)

def search_meal_cost(venue_name: str) -> str:
    query = f"{venue_name} 식대 뷔페 가격 2025 OR 2026"
    try:
        search_res = search_tool.invoke({"query": query})
        results_list = search_res.get("results", [])
        if results_list:
            snippets = [f"- {item.get('title', '')}: {item.get('content', '')}" for item in results_list]
            return "\n".join(snippets)
        return "수도권 일반 웨딩홀 평균 식대(약 75,000원-85,000원) 기준 적용"
    except Exception as e:
        return "웨딩홀 기본 표준 식대(약 80,000원) 대체 적용"

# 5. 3단계: 최종 판정 Pydantic 출력 스키마 (성의도 등급 필드 추가)
class WeddingGiftVerdict(BaseModel):
    delivery_grade: str = Field(
        description="판정된 청첩장 성의도 등급: '1등급 (직접 식사 대접)', '2등급 (개인 정성 연락)', '3등급 (모바일 링크만 살포/스팸)' 중 하나"
    )
    estimated_meal_cost: int = Field(description="조사된 예식장의 1인 식대 원가 (원 단위 정수)")
    recommended_amount: int = Field(
        description="최종 추천 축의금 금액 (원 단위 정수). 패스는 반드시 0, 불참송금은 30000 또는 50000, 직접참석 식사일 때만 식대 이상(10만 이상). 3등급은 5만 초과 절대 불가."
    )
    attendance_decision: str = Field(description="행동 지침: '직접 참석하여 식사', '식사 없이 봉투만 전달', '계좌 송금 후 불참', '정중한 축하 인사만 전송(0원)'")
    reasoning_analysis: str = Field(description="식대 원가, 관계 친밀도, 전달 성의도, 신분을 종합 분석한 현실적 판정 이유")
    kakao_message_template: str = Field(description="즉시 전송용 카톡 메시지. 참석 시 축하 및 참석 약속 멘트, 불참 시 정중한 선약 핑계와 축하 멘트")

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

3. 행동 지침과 recommended_amount 값의 100% 일치:
   - attendance_decision이 '정중한 축하 인사만 전송(0원)'이면 recommended_amount는 반드시 0
   - attendance_decision이 '계좌 송금 후 불참'이면 recommended_amount는 반드시 30,000 또는 50,000
   - attendance_decision이 '직접 참석하여 식사'일 때만 recommended_amount가 식대 원가 이상(100,000원 이상)

4. 행동 지침과 kakao_message_template 내용의 100% 일치:
   - '직접 참석하여 식사' 판정 시: "기쁜 날 직접 찾아뵙고 축하해 드리겠다"는 축하 및 참석 약속 멘트 작성
   - '불참' 판정 시: 상대방이 섭섭하지 않도록 품격 있는 선약 핑계와 따뜻한 축하 문구 작성

5. 식대 원가 하한선 원칙:
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
        return "상황을 입력해 주세요.", "대기 중", "0 원", "대기 중", "내용을 작성해 주시면 분석관이 판정표를 작성합니다.", ""
    
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
    cost_text = f"{result.estimated_meal_cost:,} 원 (1인 기준)"
    amount_text = f"{result.recommended_amount:,} 원" if result.recommended_amount > 0 else "0 원 (정중한 축하 인사 후 패스)"
    
    return grade_text, cost_text, result.attendance_decision, amount_text, result.reasoning_analysis, result.kakao_message_template

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
            out_meal_cost = gr.Textbox(label="🍽️ 조사된 예식장 1인 식대 원가")
            out_decision = gr.Textbox(label="⚖️ 최종 행동 지침 (Action Verdict)")
            out_amount = gr.Textbox(label="💰 추천 적정 축의금")
            out_reasoning = gr.TextArea(label="💡 현실주의 판정 근거 (죄책감 해소)", lines=5)
            out_kakao = gr.TextArea(label="📱 즉시 전송용 완벽 방어 카톡 멘트 (원클릭 복사)", lines=4)
            
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
