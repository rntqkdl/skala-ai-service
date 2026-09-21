import os
import re
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

# 5. 3단계: 최종 판정 Pydantic 출력 스키마 (밀도 정량 지수 및 맞춤형 카톡 멘트)
class WeddingGiftVerdict(BaseModel):
    delivery_grade: str = Field(
        description="판정된 청첩장 성의도 등급: '1등급 (직접 식사 대접)', '2등급 (개인 정성 연락)', '3등급 (모바일 링크만 살포/스팸)' 중 하나"
    )
    relationship_density_score: int = Field(
        description="인간관계 밀도 정량 점수 (0-100점). 고락 공유 경험(프로젝트 밤샘 등), 최근 교류 빈도, 정서적 부채감을 종합 평가한 수치"
    )
    density_stars: str = Field(
        description="밀도 점수 기반 별표 시각화 (예: '★★★★★ (95점) [고밀도 전우애/인생멘토]', '★★★☆☆ (50점) [비즈니스 지인형]', '★☆☆☆☆ (10점) [희석된 연락두절형]')"
    )
    venue_name: str = Field(
        description="확인된 예식장 명칭 (예: 강남 아펠가모 선릉, 신라호텔 다이너스티홀)"
    )
    meal_cost_range: str = Field(
        description="실제 식대 시세 가격대 범위. (자릿수 오류 엄금: '20만원'은 '200,000원'이며 절대 '20,000원'이 아님! 10만원 이상은 '약 200,000원 - 250,000원 선' 또는 '약 20만원 - 25만원 선'으로 올바르게 표기할 것)"
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
        description="상대방과의 관계(사수, 동기, 동창, 타 부서 등)와 수신 맥락을 반영하여 복사해 바로 보낼 수 있는 실전 카카오톡 메시지. 상대방의 위계/호칭에 따른 존댓말/구어체 완벽 매칭, 밥 대접 감사/원거리 양해/식사 미참석 양해 등 필수 반영."
    )

system_instruction = """
당신은 대한민국 2030 세대의 눈치 비용과 지갑을 지켜주는 15년 차 현실주의 경조사 에티켓 분석관입니다.
사용자가 자유롭게 적은 하소연 텍스트를 읽고 상황을 입체적으로 파악하여 냉철하고 수학적으로 판정하세요.

[자릿수 변환 절대 주의 규칙 (단위 환각 방지)]
- 한국어의 '20만원', '25만원'은 각각 200,000원과 250,000원입니다. '만' 단위를 1,000으로 착각하여 '20,000원 - 25,000원'으로 자릿수를 누락(0 하나 탈락)하는 단위 환각 오류를 절대 범하지 마세요.
- 특급 호텔(신라호텔 등) 식대는 20,000원이 아니라 200,000원대입니다. meal_cost_range에는 반드시 '1인 약 200,000원 - 250,000원 선' 또는 '1인 약 20만원 - 25만원 선'으로 올바르게 표기하세요.

[인간관계 밀도(Density) 정량화 및 별표 평가 기준 (0-100점)]
- 85점 - 100점 (★★★★★) [고밀도 전우애/인생멘토]: 시험, 부트캠프, 프로젝트 밤샘 동고동락, 멘토링 수혜, 최근 6개월 내 밀착 교류 및 밥 대접 수혜.
- 60점 - 84점 (★★★★☆) [친밀한 동료/절친]: 직속 팀원, 사수, 정기적 사적 모임 유지 친구.
- 35점 - 59점 (★★★☆☆) [비즈니스 동료/보통 지인]: 타 부서 동료, 업무상 주기적 마주침, 가끔 안부 묻는 동기.
- 15점 - 34점 (★★☆☆☆) [희석된 옛 지인]: 연차만 오래되고 최근 2-3년 이상 연락이 뜸한 동창.
- 0점 - 14점 (★☆☆☆☆) [연락두절 스팸/결례]: 4-5년 이상 연락 없다가 단톡방에 모바일 청첩장 링크만 투척한 결례 관계.

[청첩장 성의도 3대 기준 정의 및 판정 룰]
1. 성의도 등급 분류 (delivery_grade):
   - 1등급 (직접 식사 대접): 사전에 직접 만나 밥 한 끼 대접하며 종이 청첩장을 건넨 경우 -> 축의금 100,000원 - 200,000원, 직접 참석 권장
   - 2등급 (개인 정성 연락): 개인적인 안부 전화나 정성스러운 1:1 메시지와 함께 모바일 청첩장을 보낸 경우 -> 직접 참석(100,000원) 또는 송금 불참(50,000원)
   - 3등급 (모바일 링크만 살포/스팸): 수년간 연락 없다가 단톡방이나 카톡 링크만 툭 던진 경우 -> 상대방의 명백한 결례이므로 직접 식사 참석 절대 금지, '계좌 송금 후 불참(30,000원-50,000원)' 또는 '정중한 패스(0원)'로만 판정. 3등급 축의금은 절대 50,000원 초과 불가.

2. 성의도 단조 증가성 원칙 (1등급 >= 2등급 >= 3등급):
   - 3등급의 추천 축의금이 1등급이나 2등급보다 커지는 역전 현상은 절대 불가합니다.

3. 실제 예식장 식대 시세 및 가격대 제공 (meal_cost_range):
   - Tavily 검색 결과를 분석하여 해당 예식장의 실제 식대 시세 가격대 범위(예: "1인 뷔페 약 85,000원 - 92,000원 선", "1인 코스 약 200,000원 - 250,000원 선")를 구체적으로 명시하세요.
   - 기준 식대 원가(estimated_meal_cost)는 이 범위 내 대표값으로 설정합니다.

4. 행동 지침과 recommended_amount 값의 100% 일치:
   - attendance_decision이 '정중한 축하 인사만 전송(0원)'이면 recommended_amount는 반드시 0
   - attendance_decision이 '계좌 송금 후 불참'이면 recommended_amount는 반드시 30,000 또는 50,000
   - attendance_decision이 '직접 참석하여 식사'일 때만 recommended_amount가 식대 원가 이상(100,000원 이상)

5. 상대방 위계 및 상황별 실전 카톡 멘트 동적 생성 규칙 (kakao_message_template):
   - [축의금 액수 언급 절대 금지]: 카톡 메시지 본문에는 축의금이나 돈, 액수와 관련된 단어(얼마를 낸다, 준비했다 등)를 단 한마디도 쓰지 마세요. 한국 경조사 문화에서 당사자에게 축의금 금액을 직접 알리는 것은 중대한 결례입니다. 오직 진심 어린 축하, 밥 대접에 대한 감사, 당일 참석 약속(또는 불참 선약 양해) 내용만 담으세요.
   - [호칭 및 말투 격식 완벽 매칭]:
     1. 상대방이 사수, 선배, 직장 상사인 경우: 반드시 "형/선배님/대리님, 결혼 진심으로 축하드립니다"로 시작하여 정중하고 깍듯한 존댓말을 구사하세요.
     2. 상대방이 동기, 친구인 경우: 친근하고 자연스러운 구어체를 사용하세요 (예: "형/OO아, 결혼 진짜 축하해!").
     3. 상대방이 연락 없던 동창이거나 단톡방 스팸인 경우: 사적인 군더더기 없이 깔끔하고 건조한 표준 축하 인사를 작성하세요.
   - [사전 맥락 및 특수 상황 필수 반영]:
     1. 사전에 밥을 얻어먹은 경우: "지난번에 바쁘신 와중에도 맛있는 식사 대접해 주셔서 정말 감사했습니다" 문구를 본문에 반드시 포함하세요.
     2. 원거리(부산/지방)로 불참하는 경우: "예식장이 부산이라 직접 찾아뵙고 축하드리지 못해 마음이 정말 무겁고 아쉽습니다"와 같이 지리적 이동 사유를 정중히 밝히세요.
     3. 당일 선약으로 식사 없이 봉투만 전달하는 경우: "당일 미리 잡힌 선약이 있어서 식사는 함께하지 못하고, 식장 앞에서 인사드리고 봉투만 전달드려야 할 것 같습니다. 죄송하고 진심으로 축하드립니다"로 사전 양해를 구하세요.

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
        return "상황을 입력해 주세요.", "대기 중", "대기 중", "대기 중", "0 원", "내용을 작성해 주시면 분석관이 판정표를 작성합니다.", ""
    
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
    density_text = f"{result.density_stars} ({result.relationship_density_score}점 / 100점)"
    
    # 자릿수 단위 오류 방어 가드레일 (특급호텔 20,000원 -> 200,000원 0 탈락 자동 보정)
    cost_range = result.meal_cost_range
    if result.estimated_meal_cost >= 100000:
        cost_range = re.sub(r'(?<![0-9])([1-3][0-9]),000(?![0-9])', r'\g<1>0,000', cost_range)
                
    venue_price_text = f"{result.venue_name}: {cost_range} (기준 식대: {result.estimated_meal_cost:,} 원)"
    decision_text = result.attendance_decision
    amount_text = f"{result.recommended_amount:,} 원" if result.recommended_amount > 0 else "0 원 (정중한 축하 인사 후 패스)"
    
    # 카톡 메시지 에티켓 가드레일 (축의금 액수 직접 언급 문구 자동 제거)
    kakao_text = result.kakao_message_template
    kakao_text = re.sub(r"축의금[^\.\!\?]*[\.\!\?]?", "", kakao_text).strip()
    kakao_text = re.sub(r"[0-9만,]+원[^\.\!\?]*[\.\!\?]?", "", kakao_text).strip()
    kakao_text = re.sub(r"\s{2,}", " ", kakao_text)
    
    return grade_text, density_text, venue_price_text, decision_text, amount_text, result.reasoning_analysis, kakao_text

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
                placeholder="예시: SK AX 동기형 결혼식인데 신라호텔에서 해. 직접 밥 사주시면서 종이 청첩장 주셨는데, 나 아직 취준생 백수라... 축의금 얼마 내고 참석해야 하지?",
                lines=5
            )
            with gr.Row():
                btn_ex1 = gr.Button("📋 예시 1: 3등급 동창 (5년 만의 카톡 링크 테러)", size="sm")
                btn_ex2 = gr.Button("📋 예시 2: 1등급 동기 (신라호텔 밥 대접 + 취준생)", size="sm")
            with gr.Row():
                btn_ex3 = gr.Button("📋 예시 3: 2등급 원거리 (서울-부산 KTX 이동 부담)", size="sm")
                btn_ex4 = gr.Button("📋 예시 4: 2등급 동료 (서초 엘타워 식사 미참석)", size="sm")
                
            submit_btn = gr.Button("🔍 냉철한 판정 리포트 생성", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 분석관의 현실주의 판정표")
            out_grade = gr.Textbox(label="🏷️ 판정된 청첩장 성의도 등급")
            out_density = gr.Textbox(label="⭐ 인간관계 밀도 지수 (Relationship Density)")
            out_meal_cost = gr.Textbox(label="🍽️ 조사된 예식장 실제 시세 및 1인 식대 원가")
            out_decision = gr.Textbox(label="⚖️ 최종 행동 지침 (Action Verdict)")
            out_amount = gr.Textbox(label="💰 추천 적정 축의금")
            out_reasoning = gr.TextArea(label="💡 현실주의 판정 근거 (죄책감 해소)", lines=5)
            out_kakao = gr.TextArea(label="📱 상대방 맞춤형 실전 카톡 멘트 (상황 및 호칭 완벽 매칭)", lines=4)
            
    # 예시 버튼 바인딩 (현실감 넘치는 커뮤니티/실제 하소연 어조)
    btn_ex1.click(
        fn=lambda: "고등학교 졸업하고 5년 동안 연락 한 번 없던 동창이 갑자기 단톡방에 모바일 청첩장 링크만 띡 올렸어. 예식장은 강남 아펠가모 선릉이라는데, 나 지금 취준생이거든? 축하한다고 5만원 보내야 해, 아니면 그냥 읽씹하고 넘어가도 돼?",
        outputs=user_input
    )
    btn_ex2.click(
        fn=lambda: "SK AX 교육과정에서 6개월간 매일 밤새며 프로젝트 같이 한 동기형 결혼식인데, 신라호텔 다이너스티홀에서 해. 직접 만나서 비싼 밥 사주시면서 종이 청첩장 주셨거든? 나는 아직 무직 백수 취준생인데, 축의금 얼마 내고 참석해야 형한테 민폐가 안 될까?",
        outputs=user_input
    )
    btn_ex3.click(
        fn=lambda: "전 직장에서 1년 같이 일했던 동료인데 1:1 카톡으로 정중하게 모바일 청첩장 주셨어. 근데 식장이 부산 그랜드모먼트라 서울에서 KTX 왕복비만 12만원 깨지고 주말 하루가 통째로 날아가. 차비 지원 언급은 없는데, 직접 참석해야 할까 아니면 5만원만 송금할까?",
        outputs=user_input
    )
    btn_ex4.click(
        fn=lambda: "업무상 주 1회 마주치는 타 부서 대리님인데 1:1로 정중하게 청첩장 주셨어. 식장은 서초 엘타워야. 당일 선약이 있어서 식사는 안 하고 축의금 봉투만 전달하고 바로 나올 생각인데, 5만원만 내도 예의에 어긋나지 않을까?",
        outputs=user_input
    )
    
    # 제출 버튼 바인딩
    submit_btn.click(
        fn=process_wedding_inquiry,
        inputs=user_input,
        outputs=[out_grade, out_density, out_meal_cost, out_decision, out_amount, out_reasoning, out_kakao]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
