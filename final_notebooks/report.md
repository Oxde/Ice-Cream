

# Project Algorithm

## Project Report

## Applied Mathematics: Data Science

## Inholland

## Amsterdam

## June 20, 2025


## Contents

- 1 Introduction
- 2 Data Overview
- 3 Data Preprocessing
- 4 Exploratory Data Analysis (EDA)
- 5 Model Development
- 6 Model Evaluation
- 7 ROI Analysis and Budget Optimization
- 8 Conclusion and Strategic Recommendations
- 9 Appendix


## 1 Introduction

In a competitive and saturated market, companies likeBen & Jerry’saim to maximize
the return on their marketing investments. However, without proper analysis, allocating
media budgets effectively across various channels remains a significant challenge. To
address this,Marketing Mix Modeling (MMM)provides a quantitative approach
that helps determine which marketing efforts drive sales and how future spending should
be optimized.

This project was executed in collaboration withGroup M, the world’s leading media
investment group. The main objective was to create a statistical model using historical
data from Ben & Jerry’s to evaluate the effectiveness of different marketing channels and
provide actionable recommendations for budget reallocation.

The provided dataset spans from 2020 to 2024 and includes weekly information on sales,
media spend, promotional activity, and external factors. The team followed a structured
approach: cleaning and unifying the data, engineering features such as adstock and sat-
uration effects, building multiple models, and simulating budget scenarios to assess ROI
outcomes.

This report outlines the process, findings, and strategic insights derived from the model,
with the ultimate goal of helping Ben & Jerry’s improve their marketing performance
through data-driven decision-making.

## 2 Data Overview

The dataset used in this project was provided through collaboration with Group M and
represents a comprehensive view of Ben & Jerry’s weekly performance across several
marketing, environmental, and promotional dimensions. The data spans from January
2022 to early 2024, covering over 110 weeks of information at a weekly resolution.

The original data came from multiple sources and was integrated into a single unified
dataset through systematic preprocessing steps. This unified dataset includes:

- Weekly sales figures for Ben & Jerry’s (revenue).
- Media spend per channel (TV, Digital, Out-of-Home, and Print).
- Promotional activity flags (e.g., in-store promotions, promo emails).
- External environmental factors (e.g., temperature, rainfall, wind).
- Time-based features (e.g., holidays, seasonality indicators).

To ensure consistency across channels and reduce noise from sparsely populated weeks,
the data was filtered to begin from 2022. This cut-off was selected based on completeness
of media spend and the availability of weather and promotional activity data.

Each weekly observation was defined by a combination of sales output and its associated
drivers. Media spend was recorded in euros, while weather conditions were retrieved
from public meteorological data and aligned temporally with sales weeks. Promotional
indicators were binary flags reflecting the presence or absence of active campaigns in that
week.


The final modeling dataset, as used in the enhanced models and simulations, was fully
cleaned and included all relevant explanatory variables. Its structure laid the foundation
for all subsequent analysis, transformations, and budget simulations. Through this con-
solidated dataset, we were able to capture not only the direct effects of media but also
indirect seasonal and contextual drivers of sales.

## 3 Data Preprocessing

Before modeling, a thorough preprocessing pipeline was developed to clean, structure,
and enrich the dataset. This ensured that the final input to the model was not only
consistent and complete but also captured all relevant sales drivers.

Timeframe Filtering and Alignment

To ensure full availability of all features across all sources (media, promo, weather), the
dataset was filtered to include only weeks from January 2022 onward. This decision
removed earlier periods with sparse data or missing channel spend, thereby improving
overall quality and reducing the risk of biased model coefficients.

Data Integration and Merging

The final dataset was constructed by merging several input tables:

- Weekly sales and media spend (from Group M)
- Promotional indicators (promo flags, email campaigns)
- Weather data (temperature, rainfall, wind) from external sources

Data was joined on a common weekly time key. The columns were standardized using
consistent naming conventions and formatting (e.g., lowercase, snake-case), and inconsis-
tent or malformed rows were removed.

Feature Engineering

To prepare the data for modeling, several new variables were engineered:

- Time-based features:Week number, quarter, and holiday flags were derived from
    the date.
- Weather adjustments: Weekly average temperature, total rainfall, and wind
    index were introduced to capture external demand effects.
- Promotion and media indicators:A combined flag for promotional weeks and
    another for email outreach were added to isolate campaign impact.
- Spend consistency: Weeks where total spend across major channels was zero
    were excluded, as they likely reflected media inactivity or reporting gaps.

These steps enhanced the dataset’s ability to capture seasonal, promotional, and external
influences on sales.


Data Validation

Post-processing checks ensured the dataset was fit for modeling:

- Missing values were minimal and handled via removal or forward-fill depending on
    context.
- The time series was continuous with no date gaps.
- All numerical variables were checked for outliers and skewness, informing future
    transformations.

This structured preprocessing pipeline produced a unified and enriched dataset, which
served as the reliable foundation for modeling and simulation phases.

## 4 Exploratory Data Analysis (EDA)

The exploratory phase aimed to uncover sales patterns, assess media behavior, and evalu-
ate the influence of contextual variables. This analysis guided the design of the modeling
pipeline by identifying key predictors and supporting business hypotheses.

Sales and Media Trends

Weekly sales data showed moderate variability with notable peaks around promotional
periods and end-of-year holidays. Sales remained relatively stable across quarters, but
showed temporary surges that aligned with marketing activities. Among media chan-
nels, TV and Digital were the most consistently active, while Out-of-Home (OOH) and
Print had more intermittent presence. Digital spend in particular exhibited more tactical
bursts, suggesting targeted short-term campaigns.

Correlations and Media Impact Potential

A correlation matrix was constructed to assess initial linear associations:

- Digital and TV spend showed moderate positive correlation with sales.
- Print and OOH had weaker correlations, partially due to sparse investment.
- Promotional email campaigns and general promo flags both showed immediate
    short-term uplift in sales.

While these correlations provided useful direction, they were not sufficient to quantify
actual contributions, given the presence of lags, saturation, and seasonal effects.

Seasonality and Holiday Effects

Further analysis revealed clear seasonal patterns:

- Quarter 4 (Q4) consistently outperformed other quarters in sales volume, primarily
    due to holiday demand.
- Specific holiday weeks (e.g., Christmas, Easter) showed sharp temporary increases,
    regardless of media spend.


- Weather variables such as temperature and rainfall had weaker but notable influ-
    ence, particularly in colder months.

These insights justified the inclusion of time-based and holiday-related variables in the
model to capture non-marketing fluctuations in sales.

Media Channel Behavior

Each media channel exhibited distinct characteristics:

- TV spend followed a broader reach strategy with steady investment.
- Digital media was used more surgically, often peaking during promotional weeks.
- Print and OOH had fewer active weeks and minimal overlap with major campaigns.

Understanding these behavioral differences allowed for more appropriate application of
transformations such as adstock and saturation in the modeling stage.

Outlier and Anomaly Detection

Several weeks were flagged as anomalies due to unusually low or high values in either spend
or sales. These were retained for modeling but noted for robustness checks, particularly
during validation. No structural breaks were observed, and the time series maintained
continuity throughout.

Overall, the EDA provided strong empirical support for the modeling approach by clari-
fying variable relationships, seasonal trends, and campaign dynamics.

## 5 Model Development

The goal of this project was to quantify the contribution of different media channels
and external factors to weekly sales using a data-driven, interpretable model. The de-
velopment process followed a structured progression from a simple linear baseline to an
enhanced model incorporating marketing science transformations such as adstock and
saturation.

Baseline Linear Model

We began with a multiple linear regression model using raw media spend, promotional
flags, and seasonality variables as predictors. This model offered a foundational view of
variable importance but suffered from several shortcomings:

- Media variables showed signs of multicollinearity.
- The model failed to account for carryover effects and diminishing returns.
- Residuals indicated potential violations of linear assumptions.

Despite these limitations, the baseline model served as a diagnostic tool and helped
confirm which variables were statistically significant.


Adstock Transformation

To reflect the reality that media campaigns influence consumer behavior beyond a single
week, we applied an adstock transformation to all media variables. Adstock introduces
a memory effect, modeling the lagged response to media spend.

Each transformed media variable followed this recursive formula:

```
Adstockt=xt+λ·Adstockt− 1
```
whereλrepresents the decay rate, optimized via grid search based on predictive perfor-
mance.

Saturation Transformation

To address the concept of diminishing returns, we applied a saturation function to each
adstocked variable. The log-based transformation ensures that high spend values do not
unrealistically inflate sales estimates.

```
Saturated Spend = log(1 + Adstocked Spend)
```
These transformations made the model more aligned with real-world consumer behavior
and marketing theory.

Final Model Specification

The final model included:

- Saturated and adstocked media spend for each channel (TV, Digital, OOH, Print)
- Binary promotional indicators (in-store and email)
- Holiday and quarterly seasonality flags
- Weather-related variables (temperature, wind, rainfall)

The inclusion of weather and seasonality was particularly valuable in explaining baseline
sales shifts unrelated to marketing activity.

Model Fit and Properties

The enhanced model demonstrated a strong fit to the data, with improved R²and lower
prediction error. Coefficients were stable, interpretable, and directionally aligned with
business expectations. Additionally, standard regression diagnostics showed:

- Reduced multicollinearity (all VIF ¡ 5)
- Well-behaved residuals with no visible bias or autocorrelation
- No overfitting, confirmed via holdout validation

This final model provided a robust and interpretable framework for quantifying channel-
level impact and simulating future scenarios.


## 6 Model Evaluation

After developing the final model with adstock and saturation transformations, we as-
sessed its performance using both statistical diagnostics and business interpretability.
The evaluation confirmed the model’s validity and usefulness as a tool for marketing
decision-making.

Statistical Performance

The model demonstrated a solid fit with the data, explaining a significant portion of the
variation in weekly sales.

- R-squared (R²): Approximately 0.76, indicating a strong explanatory power for
    real-world marketing data.
- Root Mean Squared Error (RMSE):Substantially reduced from the baseline,
    showing closer predictions to actual sales.
- Holdout validation: Performance remained consistent on the final 15% of the
    data, confirming that the model generalizes well and does not overfit.

Diagnostic Checks

We conducted a full set of regression diagnostics to validate assumptions:

- Linearity and additivity: Satisfied after applying transformations.
- Multicollinearity: All Variance Inflation Factors (VIF) were below 5, indicating
    low risk of inflated coefficient estimates.
- Residual analysis:Residuals appeared random and homoscedastic, with no clear
    autocorrelation or skew.

These diagnostics supported the statistical reliability of the model.

Business Interpretability

Each coefficient in the model was directionally consistent with marketing expectations:

- Digital mediahad the highest marginal return and was consistently significant
    across all validation sets.
- TV spendshowed a positive but more delayed and smoothed effect due to adstock.
- Promotional flags(including email campaigns) had a strong short-term lift, par-
    ticularly in Q4 and around holidays.
- OOH and Printhad lower, sometimes negligible, impact — consistent with lower
    and more inconsistent spend patterns.
- Weather variablesadded contextual richness, improving model fit without intro-
    ducing instability.

This interpretability was key for translating the model into actionable insights for media
optimization.


Limitations and Considerations

While the model performed well, several limitations remain:

- The model is associative, not causal — coefficients indicate correlation, not pure
    attribution.
- Some variables (e.g., Print, OOH) have limited variance, which reduces statistical
    power.
- Competitive activity, macroeconomic trends, and pricing dynamics were not in-
    cluded and could enhance future iterations.

Nonetheless, the model offers a reliable foundation for strategic budget decisions and
scenario simulations.

## 7 ROI Analysis and Budget Optimization

With the final model validated, we used it to simulate channel-level returns and explore
optimized budget allocations. This phase translated statistical outputs into actionable
marketing strategies by identifying which media investments delivered the highest return
on investment (ROI) and how spend could be reallocated for improved efficiency.

Channel-Level ROI Estimation

We calculated ROI for each media channel by comparing the incremental sales attributed
by the model to the historical spend on that channel:

### ROI =

```
Incremental Sales
Channel Spend
```
The results showed a clear hierarchy in media effectiveness:

- Digital mediaconsistently produced the highest ROI. It demonstrated both re-
    sponsiveness and scalability, making it the most efficient channel.
- TVshowed moderate ROI. While less efficient in the short term, its adstock effects
    suggest long-term brand reinforcement.
- Print and Out-of-Home (OOH)had the lowest returns and were active during
    fewer weeks, indicating a potential area for cost reduction.
- Promotional emailsyielded strong returns in specific weeks, particularly when
    aligned with Q4 holiday periods.

These ROI estimates provided the foundation for the next step: budget optimization.

Scenario-Based Budget Reallocation

Using model coefficients, we conducted simulations to estimate the sales impact of hypo-
thetical changes in media allocation, holding total budget constant.


- Areallocation scenariothat shifted 10% of spend from TV and Print to Digital
    yielded a forecasted sales increase of approximately 3.5%.
- Increasing total budget by 15% without changing the mix resulted in diminishing
    marginal returns, reinforcing the importance of spend efficiency.
- Aholiday-focused scenario, aligning email and promo spend around Q4, deliv-
    ered amplified short-term sales spikes.

These simulations confirmed that efficiency gains could be achieved not only through
higher spend but through smarter allocation.

Strategic Budget Recommendations

Based on these findings, we propose the following optimizations:

- Shift budget toward Digital, especially in Q4 and promo-heavy periods.
- Maintain a presence in TV but reduce frequency to control for diminishing
    returns.
- Reduce investment in Print and OOHunless new creative or targeting strate-
    gies can justify higher returns.
- Deploy email campaigns more strategically, particularly when paired with
    in-store promotions and seasonal boosts.

These changes aim to improve overall marketing effectiveness by focusing resources on
the channels that consistently drive measurable returns.

## 8 Conclusion and Strategic Recommendations

Project Summary and Outcomes

This project applied Marketing Mix Modeling (MMM) to quantify the impact of vari-
ous media channels, promotions, and external factors on weekly sales for Ben & Jerry’s.
By integrating multiple data sources—media spend, promotional activity, weather condi-
tions, and time-based events—into a unified, structured dataset, we were able to develop
a robust model that captured both short-term effects and broader seasonal dynamics.

The modeling process evolved from a simple linear baseline to an enhanced specification
incorporating adstock and saturation transformations. This allowed us to better reflect
consumer memory and diminishing returns, ensuring that the model aligned more closely
with established marketing science. Model diagnostics confirmed strong statistical reli-
ability, while simulation outputs provided practical insights into how marketing efforts
could be optimized moving forward. The final model demonstrated that digital media,
when paired with well-timed promotional activity, offered the most consistent and scalable
returns. TV retained its role as a brand-building channel, while Print and Out-of-Home
contributed less impact and showed potential for strategic re-evaluation.


Strategic Next Steps

The findings of this analysis provide a clear path forward for improving marketing ef-
fectiveness. The first recommendation is to rebalance media spend in favor of high-
performing channels such as Digital, particularly in high-impact quarters like Q4. While
maintaining TV presence is still justified, adjustments in frequency and targeting could
improve its cost efficiency. Promotional campaigns - especially those supported by email

- should be more tightly aligned with seasonal demand patterns to capture existing con-
sumer interest at the right moment.

Furthermore, Print and OOH should either be enhanced through creative innovation or
considered for budget reallocation, given their lower return under current execution. The
model can also serve as a forecasting tool in future planning cycles, with opportunities
to extend its capabilities by incorporating pricing data, competitor activity, or more
granular targeting variables.

Overall, this project demonstrates the value of data-driven media planning and provides
a foundation for ongoing optimization of Ben & Jerry’s marketing strategy.


## 9 Appendix

## Contact Information

- Nikita Marfitsyn – 711189@student.inholland.nl
- Baraa Semsmieh – 656645@student.inholland.nl
- Mehedi Hasan – 712553@student.inholland.nl
- Joost Donders – 687268@student.inholland.nl
- Ayitey Nii-Armah – 714025@student.inholland.nl

## Acknowledgements

We would like to acknowledge the assistance of ChatGPT by OpenAI, which was used for
proofreading, and providing LaTeX support during the preparation of this project plan.
All decisions regarding content, structure, and analysis were made by the project team.


