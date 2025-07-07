Project Algorithm

Project Report

Applied Mathematics: Data Science

Inholland

Amsterdam

July 5, 2025

Contents

1

Introduction

2

2

Data Overview

2

3

Data Preprocessing

4

4

Exploratory Data Analysis \(EDA\)

6

5

Model Development

7

6

Model Evaluation

9

7

ROI Analysis and Budget Optimization

10

8

Conclusion and Recommendations

11

1

1

Introduction

In a competitive and saturated market, companies aim to maximize the return on their marketing investments. However, without proper analysis, allocating media budgets effectively across various channels remains a significant challenge. To address this, Marketing Mix Modeling \(MMM\) provides a quantitative approach that helps determine which marketing efforts drive sales and how future spending should be optimized. 

This project was executed as an independent academic study in collaboration with Group M. The main objective was to create a Marketing Mix Model \(MMM\) using market data from a Dutch ice cream company to evaluate the effectiveness of different marketing channels and provide actionable recommendations for budget optimization specifically tailored to the Netherlands market context. 

The provided dataset spans from 2020 to 2025 and includes weekly information on sales, media spend, promotional activity, and external factors including Dutch weather data and cultural calendar events. The team followed a structured approach: cleaning and unifying the data, engineering features such as adstock and saturation effects, building multiple models, and simulating budget scenarios to assess ROI outcomes. 

This report outlines the process, findings, and strategic insights derived from the model, with the ultimate goal of helping the company improve its marketing performance through data-driven decision-making. 

2

Data Overview

The dataset used in this project was provided through collaboration with Group M and represents a comprehensive view of the company’s weekly performance across several marketing, environmental, and promotional dimensions. The data originally spanned from 2020 to 2025 but was filtered to focus on January 2022 to December 2024, covering 156 weeks of information at a weekly resolution after preprocessing and filtering steps. 

The original data came from multiple sources and was integrated into a single unified dataset through systematic preprocessing steps. This unified dataset includes: 1. Promotional activity: This includes weekly indicators for in-store promotions and email campaigns. 

The promotional data was binary \(active/inactive\) and helped us model short-term lifts in sales associated with specific promotional efforts. 

2. External factors: Weather variables such as weekly average temperature, hours of sunshine, and rainfall were collected from Dutch meteorological records. These variables were used to quantify the impact of climate conditions on sales of ice cream, which is inherently sensitive to seasonal changes. 

2



3. Dutch cultural calendar features: Binary flags were included for national holidays and school vacations, particularly King’s Day and Liberation Day. Additional time-based features such as week number and seasonal indicators were also derived to capture cyclical consumer behaviors linked to cultural routines in the Netherlands. 

4. Weekly Sales: This is the primary target variable used in our Marketing Mix Modeling \(MMM\). The dataset spans over three years of weekly sales values. Sales volume shows strong seasonal variation, with higher peaks during warmer periods. 

These patterns are consistent with industry knowledge of ice cream consumption trends in temperate climates. Figure ?? shows the weekly and monthly sales trends over time, while the distribution of sales is illustrated in the appendix \(Figure 2\). 

Figure 1: Weekly and Monthly Average Sales Trends Over Time 5. Media Spend: Spend data was collected across six primary marketing channels: TV Branding, Radio \(National and Local\), Search, Social Media, and Out-of-Home \(OOH\) advertising. Each channel showed distinct temporal spending behavior, which aligned with the company’s marketing campaigns and budget priorities. 

Weekly trends and total spend are visualized in Figure 2, while the channel-wise spend distribution and overall budget allocation are presented in the appendix \(Figures 4 and 5\). 

3



Figure 2: Weekly Media Spend per Channel and Total Spend Over Time Each weekly observation was defined by a combination of sales output and its associated drivers. Media spend was recorded in euros, while weather conditions were retrieved from public meteorological data and aligned temporally with sales weeks. Promotional indicators were binary flags reflecting the presence or absence of active campaigns in that week. 

The final modeling dataset, as used in the enhanced models and simulations, was fully cleaned and included all relevant explanatory variables. Its structure laid the foundation for all subsequent analysis, transformations, and budget simulations. Through this con-solidated dataset, we were able to capture not only the direct effects of media but also indirect seasonal and contextual drivers of sales. 

3

Data Preprocessing

Before modeling, a thorough preprocessing pipeline was developed to clean, structure, and enrich the dataset. This ensured that the final input to the model was not only consistent and complete but also captured all relevant sales drivers. 

Timeframe Filtering and Alignment

As mentioned in the last section, to ensure consistency across channels and reduce noise from sparsely populated weeks, the data was filtered to focus on January 2022 to December 2024, covering exactly 156 weeks of information at a weekly resolution after preprocessing and filtering steps. This decision removed earlier periods with sparse data or 4

missing channel spend, thereby improving overall quality and reducing the risk of biased model coefficients. 

Data Integration and Merging

The final dataset was constructed by merging several input tables:

• Weekly sales and media spend \(from Group M\)

• Promotional indicators \(promo flags, email campaigns\)

• Weather data \(temperature, rainfall, wind\) from external sources Data was joined on a common weekly time key. The columns were standardized using consistent naming conventions and formatting \(e.g., lowercase, snake-case\), and inconsistent or malformed rows were removed. 

Feature Engineering

To prepare the data for modeling, several new variables were engineered:

• Time-based features: Week number, quarter, and holiday flags were derived from the date. 

• Weather adjustments: Weekly average temperature, total rainfall, and wind index were introduced to capture external demand effects. 

• Promotion and media indicators: A combined flag for promotional weeks and another for email outreach were added to isolate campaign impact. 

• Spend consistency: Weeks where total spend across major channels was zero were excluded, as they likely reflected media inactivity or reporting gaps. 

These steps enhanced the dataset’s ability to capture seasonal, promotional, and external influences on sales. 

Data Validation

Post-processing checks ensured the dataset was fit for modeling:

• Missing values were minimal and handled via removal or forward-fill depending on context. 

• The time series was continuous with no date gaps. 

• All numerical variables were checked for outliers and skewness, informing future transformations. 

This structured preprocessing pipeline produced a unified and enriched dataset, which served as the reliable foundation for modeling and simulation phases. 

5



4

Exploratory Data Analysis \(EDA\)

The exploratory phase aimed to uncover sales patterns, assess media behavior, and evaluate the influence of contextual variables. This analysis guided the design of the modeling pipeline by identifying key predictors and supporting business hypotheses. 

Correlations and Media Impact Potential A preliminary correlation analysis was performed to evaluate the linear associations between media spend and sales. As shown in Figure 3, digital and TV showed moderate positive relationships with sales, indicating their potential influence. Print and OOH

displayed weaker associations, which could be attributed to low variation in spend or limited activation weeks. Promotional emails and in-store flags exhibited stronger short-term connections to sales, hinting at their temporal impact. However, correlation alone was insufficient to infer true impact, as it did not account for timing delays, indirect effects, or diminishing returns. 

Figure 3: Correlation Between Sales and Media Channels 6

Sales and Media Trends

Weekly sales data displayed stable yet cyclical behavior, with noticeable peaks during promotional events and the holiday season, particularly in Q4. Media spend varied across channels. TV and Digital had the most consistent investment patterns, with TV exhibit-ing a broad, steady presence while Digital appeared more tactical and burst-oriented. 

OOH and Print spend were more erratic and often inactive in several weeks, indicating less consistent use as strategic channels. 

Seasonality and Holiday Effects

The analysis revealed clear seasonality patterns. Sales consistently increased during Q4, suggesting the influence of holiday-related consumer demand. Holidays such as Christ-mas, New Year, and Easter were associated with temporary spikes in sales, even when media investment remained flat. This justified the inclusion of holiday and quarter-based dummy variables in the model. Weather variables such as temperature and rainfall were also explored. Although their individual correlation with sales was weak, they contributed to explaining sales variation, particularly during extreme seasonal weeks. 

Media Channel Behavior

Each media channel demonstrated unique behavioral patterns. TV spend was character-ized by its long-term, continuous nature, suggesting a role in brand maintenance rather than immediate sales response. 

Digital spend, on the other hand, was sharply con-centrated around promotional events, aligning well with short-term lifts in performance. 

OOH and Print were used sporadically and often without overlap with major sales weeks, limiting their explanatory power. These distinct activation styles highlighted the need for transformation techniques like adstock and saturation, which were later incorporated into the model to reflect media lag and diminishing returns. 

Outlier and Anomaly Detection

The dataset included several weeks with unusually high or low values in either media spend or sales. These outliers were not removed but were documented and tested for robustness during model validation. No major structural breaks or missing periods were found in the time series. The overall continuity and richness of the data provided a strong basis for reliable modeling. 

5

Model Development

The goal of this project was to quantify the contribution of different media channels and external factors to weekly sales using a data-driven, interpretable model. The development process followed a structured progression from a simple linear baseline to an enhanced model incorporating marketing science transformations such as adstock and saturation. 

7

Baseline Linear Model

We began with a multiple linear regression model using raw media spend, promotional flags, and seasonality variables as predictors. This model offered a foundational view of variable importance but suffered from several shortcomings:

• Media variables showed signs of multicollinearity. 

• The model failed to account for carryover effects and diminishing returns. 

• Residuals indicated potential violations of linear assumptions. 

Despite these limitations, the baseline model served as a diagnostic tool and helped confirm which variables were statistically significant. 

Adstock Transformation

To reflect the reality that media campaigns influence consumer behavior beyond a single week, we applied an adstock transformation to all media variables. Adstock introduces a memory effect, modeling the lagged response to media spend. 

Each transformed media variable followed this recursive formula: Adstockt = xt \+ λ · Adstockt−1

where λ represents the decay rate, optimized via grid search based on predictive performance. 

Saturation Transformation

To address the concept of diminishing returns, we applied a saturation function to each adstocked variable. The log-based transformation ensures that high spend values do not unrealistically inflate sales estimates. 

Saturated Spend = log\(1 \+ Adstocked Spend\) These transformations made the model more aligned with real-world consumer behavior and marketing theory. 

Final Model Specification

The final model included:

• Saturated and adstocked media spend for each channel \(TV, Digital, OOH, Print\)

• Binary promotional indicators \(in-store\)

• Holiday and quarterly seasonality flags

• Weather-related variables \(temperature, wind, rainfall\) The inclusion of weather and seasonality was particularly valuable in explaining baseline sales shifts unrelated to marketing activity. 

8

6

Model Evaluation

Evaluation Methodology

To ensure a realistic and unbiased evaluation of model performance, we applied a temporal train/test split. This approach reflects how MMM would be used in practice for forecasting future outcomes without contaminating results with future information. 

The first 129 weeks of data \(approximately 83%\) were used for training the model, including model fitting and hyperparameter tuning. The final 27 weeks \(17%\) were held out for testing, enabling evaluation of the model’s predictive accuracy on unseen data. This split maintained strict chronological order, eliminating any possibility of data leakage or look-ahead bias. 

Statistical Performance

The model demonstrated solid predictive performance within realistic bounds for marketing mix modeling. The best-performing variant, specialized for Dutch seasonality, achieved a test R-squared \(R2\) of 52.6%, indicating that the model explained over half of the variance in sales on previously unseen data. 

Across the different model stages, we observed progressive improvements: The baseline linear model achieved a test R2 of 45.1%, providing a basic understanding of variable impact without any media transformations. Incorporating adstock and saturation into an enhanced model raised the test R2 to 47.6%, reflecting better fit due to behavioral realism. Finally, integrating Dutch-specific seasonal flags \(holidays, weather adjustments\) resulted in a 52.6% test R2, a 16.6% improvement over the baseline. 

Training scores remained close to test scores, with the final model reaching 54.2% on training data, suggesting minimal overfitting. These results confirm that our modeling framework was statistically sound and capable of generalizing across time, enabling its use for ROI estimation and future budget planning. 

Diagnostic Checks

We conducted a full set of regression diagnostics to validate assumptions:

• Linearity and additivity: Satisfied after applying transformations. 

• Multicollinearity: All Variance Inflation Factors \(VIF\) were below 5, indicating low risk of inflated coefficient estimates. 

• Residual analysis: Residuals appeared random and homoscedastic, with no clear autocorrelation or skew. 

These diagnostics supported the statistical reliability of the model. 

Business Interpretability

Each coefficient in the model was directionally consistent with marketing expectations: 9

• Digital media had the highest marginal return and was consistently significant across all validation sets. 

• TV spend showed a positive but more delayed and smoothed effect due to adstock. 

• Promotional flags \(including email campaigns\) had a strong short-term lift, particularly in Q4 and around holidays. 

• OOH and Print had lower, sometimes negligible, impact — consistent with lower and more inconsistent spend patterns. 

• Weather variables added contextual richness, improving model fit without introducing instability. 

This interpretability was key for translating the model into actionable insights for media optimization. 

7

ROI Analysis and Budget Optimization

With the final model validated, we used it to simulate channel-level returns and explore optimized budget allocations. This phase translated statistical outputs into actionable marketing strategies by identifying which media investments delivered the highest return on investment \(ROI\) and how spend could be reallocated for improved efficiency. 

Channel-Level ROI Estimation

We calculated ROI for each media channel by comparing the incremental sales attributed by the model to the historical spend on that channel: Incremental Sales

ROI = Channel Spend

The results showed a clear hierarchy in media effectiveness. Table 1 summarizes the ROI performance across all media channels, clearly indicating which channels added value and which led to losses. 

Channel

Weekly Spend

Return per e100

Performance

Search Marketing

e622

e2,009

Top Performer

Social Media

e608

e1,366

Top Performer

TV Promo

e3,123

e983

Top Performer

TV Branding

e5,491

e23

Moderate

Radio National

e1,469

e–543

Underperforming

Radio Local

e1,863

e–858

Underperforming

Out-of-Home \(OOH\)

e793

e–1,486

Underperforming

Overall Media Efficiency

Current Overall ROI

e122 per e100

Total Weekly Media-Driven Sales

e1.7M

Optimization Potential

\+20% \( e17.7M/year\)

Table 1: Channel ROI Performance Summary 10



Figure 4: Weekly Media Spend per Channel and Total Spend Over Time These ROI estimates provided the foundation for the budget optimization. 

Scenario-Based Budget Reallocation

Using model coefficients, we conducted simulations to estimate the sales impact of hypo-thetical changes in media allocation, holding total budget constant. 

• A reallocation scenario that shifts 10% of spend from TV and Print to Digital yielded a forecasted sales increase of approximately 3.5%. 

• Increasing total budget by 15% without changing the mix resulted in diminishing marginal returns, reinforcing the importance of spend efficiency. 

• A holiday-focused scenario, aligning email and promo spend around Q4, delivered amplified short-term sales spikes. 

These simulations confirmed that efficiency gains could be achieved not only through higher spend but through smarter allocation. 

8

Conclusion and Recommendations

Project Summary

This project applied Marketing Mix Modeling \(MMM\) to quantify the impact of various media channels, promotions, and external factors on weekly sales for the company. By integrating multiple data sources - media spend, promotional activity, weather conditions, and national events - into a unified, structured dataset, we were able to develop a robust model that captured both short-term effects and broader seasonal dynamics. 

The modeling process evolved from a simple linear baseline to an enhanced specification incorporating adstock and saturation transformations. This allowed us to better reflect consumer memory and diminishing returns, ensuring that the model aligned more closely 11

with established marketing science. The model diagnostics confirmed strong statistical reliability, while the simulation outputs provided practical insights into how marketing efforts could be optimized moving forward. The final model demonstrated that digital media, when paired with well-timed promotional activity, offered the most consistent and scalable returns. TV retained its role as a brand-building channel, while Print and Out-of-Home contributed less impact and showed potential for strategic re-evaluation. 

Recommended Next Steps

Building on the insights gained from the MMM analysis, we identified several data-backed strategies to improve marketing efficiency and revenue impact within the Dutch market. 

1. Scale Up High-ROI Channels

The model revealed that some channels significantly outperform others in terms of ROI:

• Increase investment in Search marketing, which achieved the highest ROI. 

• Expand Social Media efforts to capture cost-efficient conversions. 

• Optimize the timing of TV promotions by aligning with seasonal or cultural events. 

2. Reduce Underperforming Spend

Some traditional channels delivered low or negative ROI and should be strategically reduced or restructured:

• Cut spending on Radio \(national and local\) due to consistently poor return. 

• Minimize investment in Out-of-Home \(OOH\) unless paired with a new creative or hyper-local targeting strategy. 

3. Implement Weather-Responsive Campaign Triggers The chart titled “Monthly Average Sales Trend” highlights a clear seasonal sales pattern, with noticeable peaks between May and August. This upward trend aligns with the summer months in the Netherlands, suggesting that warmer weather significantly influences consumer purchasing behavior—particularly relevant for products like ice cream. 

To capitalize on this trend, we recommend the following weather-responsive strategies:

• Trigger campaigns during heatwaves: Launch targeted campaigns when weekly temperatures exceed 25°C. These conditions correlate closely with the sharpest sales peaks visible in June, July, and August. These “heatwave windows” represent ideal periods for increased media activity to capture impulse purchases. 

• Boost investment in warm lead-up months: Intensify promotional activity during milder warm periods \(18–25°C\), especially in April through June. The graph shows that sales begin rising as early as April, and additional marketing during this ramp-up phase can further lift performance. 

Incorporating real-time temperature thresholds into media planning allows for a more agile marketing strategy. The seasonal pattern seen in the data strongly suggests that temperature is a key external driver, with statistically significant influence on weekly sales performance. Aligning campaign intensity with weather conditions ensures your marketing efforts are both timely and demand-driven. 

12



4. Align with the Dutch Cultural Calendar Figure 5 reveals a clear uplift in average sales during holiday periods compared to regular times. Sales during holidays reach an average of 145,942 units, significantly higher than the 132,461 units observed during regular periods. This suggests that Dutch holidays and culturally significant moments serve as powerful demand triggers. 

Figure 5: Sales: Holiday vs Regular Periods To harness this effect, we recommend integrating key cultural events into your campaign calendar:

• Capitalize on spring holidays: Focus campaign efforts around King’s Day \(April 27\) and Liberation Day \(May 5\). These holidays precede the summer upswing and are deeply embedded in Dutch culture, making them ideal for high-visibility, emotionally resonant campaigns. 

• Maximize summer break exposure: The months of July and August consistently deliver peak performance, as supported by both holiday-related and seasonal data. This period combines leisure time, vacation spending, and warm weather making it the perfect window for intensified marketing efforts. 

The data validates that aligning campaign timing with culturally significant and leisure-rich periods boosts sales outcomes. By embedding Dutch cultural timing into planning cycles, campaigns become not just relevant, but also better positioned to capture elevated consumer attention and maximize return on investment. 

13


# Document Outline

+ Introduction 
+ Data Overview 
+ Data Preprocessing 
+ Exploratory Data Analysis \(EDA\) 
+ Model Development 
+ Model Evaluation 
+ ROI Analysis and Budget Optimization 
+ Conclusion and Recommendations



