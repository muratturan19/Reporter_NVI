# -*- coding: utf-8 -*-
"""
Araştırmacı Ajan - ReAct Pattern ile
Anthropic Claude kullanarak web araştırması yapar
"""

import asyncio
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import logging

logger = logging.getLogger(__name__)

# Araştırma promptları
RESEARCH_SYSTEM_PROMPT = """
Sen deneyimli bir araştırmacısın. Verilen konu hakkında kapsamlı araştırma yapmak için web arama sorgularını oluşturuyorsun.

Arama sorguları oluştururken:
1. Konunun farklı yönlerini kapsa (temel özellikler, gerçek uygulamalar, teknik mimari)
2. Konuyla ilgili spesifik teknik terimleri içersin
3. Güncel bilgi için yıl belirteci ekle (örn: "2024", "2025")
4. Benzer teknolojilerle karşılaştırma ara
5. Hem resmi dokümantasyon hem de pratik örnekler ara

Sorgular:
- Genel sonuçları önlemek için yeterince spesifik
- Detaylı uygulama bilgisi almak için yeterince teknik
- Tüm konuyu kapsamak için yeterince çeşitli
- Güvenilir kaynakları hedefleyen (dokümantasyon, teknik bloglar, akademik makaleler)

Türkçe içerikler için Türkçe anahtar kelimeler de kullan.
"""

RESEARCH_HUMAN_PROMPT = """
Konu: {topic}

Bu konu için {number_of_queries} adet web arama sorgusu oluştur.
Her sorgu farklı bir açıdan konuya yaklaşsın.
"""

class ResearcherAgent:
    """Araştırmacı Ajan Sınıfı"""
    
    def __init__(self, llm, search_tool):
        self.llm = llm
        self.search_tool = search_tool
        self.llm_with_tools = llm.bind_tools([search_tool])
        
        # Prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RESEARCH_SYSTEM_PROMPT),
            ("human", RESEARCH_HUMAN_PROMPT)
        ])
        
        # Graf oluştur
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """ReAct pattern ile araştırma grafını oluştur"""
        
        async def call_model(state: Dict[str, Any]):
            """Modeli çağır"""
            logger.info("Model çağrılıyor...")
            
            messages = state.get("messages", [])
            
            # İlk çağrıda prompt oluştur
            if not messages:
                prompt_messages = self.prompt.format_messages(
                    topic=state["topic"],
                    number_of_queries=state.get("number_of_queries", 3)
                )
                messages.extend(prompt_messages)
            
            # Son mesajı kontrol et - araç çağrısı var mı?
            if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
                # Araç sonuçlarını bekle, yeni model çağrısı yap
                response = await self.llm.ainvoke(messages)
            else:
                # İlk model çağrısı - araç bağlı model kullan
                response = await self.llm_with_tools.ainvoke(messages)
            
            return {"messages": messages + [response]}
        
        async def execute_tools(state: Dict[str, Any]):
            """Araç çağrılarını çalıştır"""
            logger.info("Araçlar çalıştırılıyor...")
            
            messages = state["messages"]
            last_message = messages[-1]
            
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
            
            tool_results = []
            for tool_call in last_message.tool_calls:
                logger.info(f"Araç çalıştırılıyor: {tool_call['name']}")
                
                if tool_call['name'] == 'search_web':
                    # Araç argümanlarını parse et
                    import json

                    raw_args = tool_call.get('args', {})

                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif isinstance(raw_args, str):
                        raw_args = raw_args.strip()
                        if raw_args:
                            try:
                                parsed_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Araç argümanları JSON olarak parse edilemedi, string kullanılacak: %s",
                                    raw_args,
                                )
                                parsed_args = raw_args
                            if isinstance(parsed_args, dict):
                                args = parsed_args
                            elif isinstance(parsed_args, list):
                                args = {"queries": parsed_args}
                            else:
                                args = {"queries": [parsed_args]}
                        else:
                            args = {}
                    else:
                        logger.warning(
                            "Araç argümanları beklenmeyen tipte: %s", type(raw_args).__name__
                        )
                        args = {}

                    # Web araması yap
                    result = await self.search_tool.ainvoke(args)
                    
                    tool_results.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call['id']
                        )
                    )
            
            return {"messages": messages + tool_results}
        
        def should_continue(state: Dict[str, Any]):
            """Devam edip etmemeye karar ver"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # Son mesaj araç çağrısı içeriyorsa, araçları çalıştır
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            else:
                # Araştırma tamamlandı
                return "end"
        
        # Graf oluştur
        workflow = StateGraph(dict)
        
        # Düğümler
        workflow.add_node("model", call_model)
        workflow.add_node("tools", execute_tools)
        
        # Başlangıç
        workflow.add_edge(START, "model")
        
        # Koşullu kenarlar
        workflow.add_conditional_edges(
            "model",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Araçlardan modele dön
        workflow.add_edge("tools", "model")
        
        return workflow.compile()
    
    async def research(self, topic: str, number_of_queries: int = None):
        """Konu hakkında araştırma yap"""
        if number_of_queries is None:
            from report_agent_setup import DEFAULT_SEARCH_QUERIES
            number_of_queries = DEFAULT_SEARCH_QUERIES
            
        logger.info(f"Araştırma başlatılıyor: {topic} ({number_of_queries} sorgu)")
        
        state = {
            "topic": topic,
            "number_of_queries": number_of_queries,
            "messages": []
        }
        
        result = await self.graph.ainvoke(state)
        
        logger.info("Araştırma tamamlandı")
        return result

# Test fonksiyonu
async def test_researcher():
    """Araştırmacı ajanını test et"""
    from report_agent_setup import create_llm, search_web
    
    # Model ve araçları yükle
    llm = create_llm()
    
    # Araştırmacı oluştur
    researcher = ResearcherAgent(llm, search_web)
    
    # Test araştırması
    result = await researcher.research(
        topic="Yapay zeka ajanlarının sağlık sektöründeki uygulamaları",
        number_of_queries=3
    )
    
    print("\n=== ARAŞTIRMA SONUÇLARI ===")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"İNSAN: {message.content}")
        elif isinstance(message, AIMessage):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"AI (Araç Çağrısı): {len(message.tool_calls)} araç çağrısı")
            else:
                print(f"AI: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"ARAÇ: {message.content[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_researcher())