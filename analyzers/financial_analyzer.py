import re
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FinancialMetric:
    name: str
    value: float
    unit: str = "USD"
    source: str = ""
    confidence: float = 1.0

@dataclass
class FinancialRatio:
    name: str
    value: float
    formula: str
    interpretation: str
    is_concerning: bool = False
    concern_reason: str = ""

class FinancialAnalyzer:
    def __init__(self):
        # Common financial terms and their variations
        self.revenue_terms = [
            "revenue", "sales", "turnover", "income", "gross receipts",
            "net sales", "total revenue", "operating revenue"
        ]
        
        self.asset_terms = [
            "total assets", "assets", "current assets", "fixed assets",
            "property plant equipment", "ppe"
        ]
        
        self.liability_terms = [
            "liabilities", "total liabilities", "current liabilities",
            "long-term debt", "debt", "obligations"
        ]
        
        self.inventory_terms = [
            "inventory", "inventories", "stock", "finished goods",
            "raw materials", "work in process"
        ]
        
        # Industry standard ratios for comparison
        self.industry_standards = {
            "current_ratio": {"min": 1.0, "max": 3.0, "ideal": 1.5},
            "inventory_turnover": {"min": 4, "max": 12, "ideal": 8},
            "debt_to_equity": {"min": 0, "max": 2.0, "ideal": 1.0},
            "inventory_to_revenue": {"min": 0.05, "max": 0.25, "ideal": 0.15}
        }
    
    def extract_financial_metrics(self, text: str) -> Dict[str, List[FinancialMetric]]:
        """Extract financial metrics from text"""
        metrics = {
            "revenue": [],
            "assets": [],
            "liabilities": [],
            "inventory": [],
            "other": []
        }
        
        # Pattern to find monetary values
        money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:[MBK]|\s*(?:million|billion|thousand))?'
        
        # Split text into sentences for context
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Find all money values in sentence
            money_matches = re.findall(money_pattern, sentence, re.IGNORECASE)
            
            for money in money_matches:
                value = self._parse_money_value(money)
                
                # Determine metric type based on context
                if any(term in sentence_lower for term in self.revenue_terms):
                    metrics["revenue"].append(
                        FinancialMetric("Revenue", value, source=sentence.strip())
                    )
                elif any(term in sentence_lower for term in self.asset_terms):
                    metrics["assets"].append(
                        FinancialMetric("Assets", value, source=sentence.strip())
                    )
                elif any(term in sentence_lower for term in self.liability_terms):
                    metrics["liabilities"].append(
                        FinancialMetric("Liabilities", value, source=sentence.strip())
                    )
                elif any(term in sentence_lower for term in self.inventory_terms):
                    metrics["inventory"].append(
                        FinancialMetric("Inventory", value, source=sentence.strip())
                    )
                else:
                    # Try to extract metric name from context
                    metric_name = self._extract_metric_name(sentence, money)
                    metrics["other"].append(
                        FinancialMetric(metric_name, value, source=sentence.strip())
                    )
        
        return metrics
    
    def calculate_ratios(self, metrics: Dict[str, List[FinancialMetric]]) -> List[FinancialRatio]:
        """Calculate financial ratios from extracted metrics"""
        ratios = []
        
        # Get the most recent/highest confidence values
        revenue = self._get_best_metric(metrics.get("revenue", []))
        assets = self._get_best_metric(metrics.get("assets", []))
        liabilities = self._get_best_metric(metrics.get("liabilities", []))
        inventory = self._get_best_metric(metrics.get("inventory", []))
        
        # Current Ratio
        if assets and liabilities:
            current_ratio = assets / liabilities
            ratio = FinancialRatio(
                name="Current Ratio",
                value=current_ratio,
                formula="Current Assets / Current Liabilities",
                interpretation=f"Company has ${current_ratio:.2f} in assets for every $1 in liabilities"
            )
            
            # Check if concerning
            standards = self.industry_standards["current_ratio"]
            if current_ratio < standards["min"]:
                ratio.is_concerning = True
                ratio.concern_reason = "Low liquidity - may struggle to meet short-term obligations"
            elif current_ratio > standards["max"]:
                ratio.is_concerning = True
                ratio.concern_reason = "Excess liquidity - may indicate inefficient use of assets"
            
            ratios.append(ratio)
        
        # Inventory to Revenue Ratio
        if inventory and revenue:
            inv_to_rev = inventory / revenue
            ratio = FinancialRatio(
                name="Inventory to Revenue",
                value=inv_to_rev,
                formula="Inventory / Annual Revenue",
                interpretation=f"Inventory represents {inv_to_rev*100:.1f}% of annual revenue"
            )
            
            # Check if concerning (like CFC case)
            standards = self.industry_standards["inventory_to_revenue"]
            if inv_to_rev > standards["max"]:
                ratio.is_concerning = True
                ratio.concern_reason = f"Unusually high inventory levels - potential obsolescence risk"
                
                # Check for extreme inventory levels
                if inv_to_rev > 0.08:  # 8% is very high for most industries
                    ratio.concern_reason = "CRITICAL: Extremely high inventory levels - potential obsolescence or demand issues"
            
            ratios.append(ratio)
        
        # Inventory Turnover
        if inventory and revenue:
            # Approximate COGS as 60% of revenue if not available
            cogs = revenue * 0.6
            turnover = cogs / inventory
            ratio = FinancialRatio(
                name="Inventory Turnover",
                value=turnover,
                formula="Cost of Goods Sold / Average Inventory",
                interpretation=f"Inventory turns over {turnover:.1f} times per year"
            )
            
            standards = self.industry_standards["inventory_turnover"]
            if turnover < standards["min"]:
                ratio.is_concerning = True
                ratio.concern_reason = "Slow inventory turnover - risk of obsolescence"
            
            ratios.append(ratio)
        
        return ratios
    
    def detect_concentration_risk(self, text: str) -> List[Dict[str, Any]]:
        """Detect customer or supplier concentration risks"""
        risks = []
        
        # Patterns to find concentration
        concentration_patterns = [
            r'(\d+)%\s+of\s+(?:revenue|sales|business)',
            r'(?:represents?|accounts?\s+for|comprises?)\s+(\d+)%',
            r'single\s+customer\s+(?:represents?|accounts?\s+for)',
            r'(?:sole|only|exclusive)\s+(?:supplier|customer|client)',
            r'100%\s+(?:reliant|dependent|reliance)'
        ]
        
        for pattern in concentration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Try to extract percentage
                pct_match = re.search(r'(\d+)%', match.group())
                if pct_match:
                    percentage = int(pct_match.group(1))
                    if percentage > 20:  # Significant concentration
                        risk = {
                            "type": "concentration",
                            "severity": "high" if percentage > 50 else "medium",
                            "percentage": percentage,
                            "context": context.strip(),
                            "finding": f"{percentage}% concentration detected"
                        }
                        
                        # Special case for 100% concentration (like CFC)
                        if percentage == 100:
                            risk["severity"] = "critical"
                            risk["finding"] = "CRITICAL: 100% customer/supplier concentration"
                        
                        risks.append(risk)
        
        return risks
    
    def analyze_financial_health(self, text: str) -> Dict[str, Any]:
        """Complete financial analysis of document"""
        # Extract metrics
        metrics = self.extract_financial_metrics(text)
        
        # Calculate ratios
        ratios = self.calculate_ratios(metrics)
        
        # Detect concentration risks
        concentration_risks = self.detect_concentration_risk(text)
        
        # Generate summary
        concerning_ratios = [r for r in ratios if r.is_concerning]
        
        health_score = "healthy"
        if len(concerning_ratios) > 0:
            health_score = "warning"
        if len(concerning_ratios) > 2 or any(r.severity == "critical" for r in concentration_risks):
            health_score = "critical"
        
        return {
            "metrics": metrics,
            "ratios": ratios,
            "concentration_risks": concentration_risks,
            "health_score": health_score,
            "concerning_items": concerning_ratios + concentration_risks,
            "summary": self._generate_summary(metrics, ratios, concentration_risks)
        }
    
    def _parse_money_value(self, money_str: str) -> float:
        """Convert money string to float value"""
        # Remove $ and commas
        value_str = money_str.replace('$', '').replace(',', '')
        
        # Handle millions, billions, thousands
        multiplier = 1
        if 'M' in value_str or 'million' in value_str.lower():
            multiplier = 1_000_000
            value_str = re.sub(r'[Mm]illion', '', value_str)
            value_str = value_str.replace('M', '')
        elif 'B' in value_str or 'billion' in value_str.lower():
            multiplier = 1_000_000_000
            value_str = re.sub(r'[Bb]illion', '', value_str)
            value_str = value_str.replace('B', '')
        elif 'K' in value_str or 'thousand' in value_str.lower():
            multiplier = 1_000
            value_str = re.sub(r'[Kk]|thousand', '', value_str)
        
        try:
            return float(value_str.strip()) * multiplier
        except:
            return 0
    
    def _get_best_metric(self, metrics: List[FinancialMetric]) -> float:
        """Get the most reliable metric value from list"""
        if not metrics:
            return None
        
        # Sort by confidence and return highest
        sorted_metrics = sorted(metrics, key=lambda x: x.confidence, reverse=True)
        return sorted_metrics[0].value
    
    def _extract_metric_name(self, sentence: str, money_value: str) -> str:
        """Try to extract metric name from sentence context"""
        # Remove the money value to find what it refers to
        sentence_clean = sentence.replace(money_value, "")
        
        # Common patterns
        patterns = [
            r'(\w+(?:\s+\w+)?)\s+(?:is|was|were|of|:)',
            r'(?:total|net|gross)\s+(\w+)',
            r'(\w+)\s+(?:amounted|totaled|reached)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence_clean, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return "Financial Metric"
    
    def _generate_summary(self, metrics: Dict, ratios: List, risks: List) -> str:
        """Generate executive summary of financial analysis"""
        summary_parts = []
        
        # Revenue summary
        if metrics.get("revenue"):
            rev = self._get_best_metric(metrics["revenue"])
            summary_parts.append(f"Revenue: ${rev:,.0f}")
        
        # Concerning ratios
        concerning = [r for r in ratios if r.is_concerning]
        if concerning:
            summary_parts.append(f"Warning: {len(concerning)} concerning financial ratios detected")
        
        # Critical risks
        critical_risks = [r for r in risks if r.get("severity") == "critical"]
        if critical_risks:
            summary_parts.append(f"CRITICAL: {critical_risks[0]['finding']}")
        
        return " | ".join(summary_parts) if summary_parts else "Financial analysis complete"