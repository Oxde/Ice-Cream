# MINOR CHANGES - Sections 1 & 2

## Section 1: Introduction ✅ **KEEP WITH MINOR FIXES**

### **What to Change:**
**Current:** "July 5, 2025"
**Change to:** Current date (December 2024 or when report is finalized)

**Current:** General timeframe mention
**Add:** Brief mention that methodology was refined through iterative analysis

### **Updated Introduction Section:**

---

**Project Algorithm**

**Project Report**

**Applied Mathematics: Data Science**

**Inholland**

**Amsterdam**

**December 2024**

---

**Section 1: Introduction**

In a competitive and saturated market, companies aim to maximize the return on their marketing investments. However, without proper analysis, allocating media budgets effectively across various channels remains a significant challenge. To address this, Marketing Mix Modeling (MMM) provides a quantitative approach that helps determine which marketing efforts drive sales and how future spending should be optimized.

This project was executed as an independent academic study in collaboration with Group M. The main objective was to create a Marketing Mix Model (MMM) using market data from a Dutch ice cream company to evaluate the effectiveness of different marketing channels and provide actionable recommendations for budget optimization specifically tailored to the Netherlands market context.

The provided dataset spans from 2020 to 2025 and includes weekly information on sales, media spend, promotional activity, and external factors including Dutch weather data and cultural calendar events. The team followed a structured approach: cleaning and unifying the data, engineering features through **data-driven optimization of adstock and saturation effects**, building multiple models with **rigorous validation**, and simulating budget scenarios to assess ROI outcomes through **counterfactual analysis**.

This report outlines the **corrected methodology**, findings, and strategic insights derived from the model, with the ultimate goal of helping the company improve its marketing performance through data-driven decision-making.

---

## Section 2: Data Overview ✅ **MINOR UPDATES NEEDED**

### **What to Change:**
1. **Add specific correlation findings**
2. **Clarify channel separation rationale**
3. **Update media channel description**

### **Changes Needed:**

**Current:** "Media Spend: Spend data was collected across six primary marketing channels: TV Branding, Radio (National and Local), Search, Social Media, and Out-of-Home (OOH) advertising."

**Change to:** 
"Media Spend: Spend data was collected across seven distinct marketing channels: TV Branding, TV Promo, Radio National, Radio Local, Search, Social Media, and Out-of-Home (OOH) advertising. **Initial correlation analysis revealed that channel correlations were below 0.7 (e.g., TV Branding ↔ TV Promo: 0.096, Radio National ↔ Radio Local: -0.064), confirming that channels should be analyzed separately rather than aggregated.**"

**Add new paragraph after media spend description:**

"**Channel Independence Analysis:** A critical finding during preprocessing was that media channels showed low inter-correlation, with most correlations below 0.3. This confirmed that each channel operates independently and should be modeled separately, contrary to common industry practices that aggregate similar channels. This independence allowed for more granular optimization and accurate attribution of each channel's contribution to sales performance."

### **Updated Section 2 Key Points:**
- ✅ Keep all existing data source descriptions
- ✅ Keep timeframe details (156 weeks, 2022-2024)
- ✅ Keep promotional and weather variable descriptions
- ➕ **Add:** Channel correlation findings
- ➕ **Add:** Rationale for separate channel analysis
- ➕ **Update:** Media channel count and naming consistency

### **Minor Formatting Updates:**
- Ensure consistent channel naming throughout (match FINAL_MODEL2.py naming)
- Add emphasis on data-driven methodology approach
- Maintain all existing content structure and quality 