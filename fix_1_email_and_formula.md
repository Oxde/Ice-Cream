### Client Feedback:
> "Maybe I did not get it correctly, but wasn't e-mail removed from the model for reasons we discussed before?"

---

## WHAT NEEDS TO BE FIXED:

### 1. INCORRECT TEXT TO CHANGE:

**Current text in Section 5 (WRONG):**
```
- Binary promotional indicators (in-store and email)
```

**Should be changed to:**
```
- Binary promotional indicators (in-store)
```

**Reason:** Email was excluded from the final model due to data availability (only 2022-2023), so we cannot mention "email" as part of the model.

---

### 2. MISSING INFORMATION TO ADD:

**Problem:** The report lacks the exact mathematical formula of our final model.

**Solution:** Add this complete model specification to Section 5:

```markdown
The final model specification is as follows:

```
Sales_t = β₀ + 
          β₁ × log(1 + Adstock(Search_spend_t)) +
          β₂ × log(1 + Adstock(TV_branding_spend_t)) +
          β₃ × log(1 + Adstock(Social_spend_t)) +
          β₄ × log(1 + Adstock(OOH_spend_t)) +
          β₅ × log(1 + Adstock(Radio_national_spend_t)) +
          β₆ × log(1 + Adstock(Radio_local_spend_t)) +
          β₇ × log(1 + Adstock(TV_promo_spend_t)) +
          β₈ × Promotion_type_t +
          β₉ × holiday_period_t +
          β₁₀ × month_sin_t + β₁₁ × month_cos_t +
          β₁₂ × week_sin_t + β₁₃ × week_cos_t +
          β₁₄ × weather_temperature_mean_t +
          β₁₅ × weather_sunshine_duration_t +
          ε_t
```

Where:
- β₀: The baseline sales intercept
- Media Predictors: Saturated and adstocked spend for 7 media channels (Search, TV Branding, Social, OOH, Radio National, Radio Local, TV Promo)
- Promotion_type: In-store promotional activities (Buy One Get One, Limited Time Offer, Price Discount)
- Holiday & Seasonality: Binary flag for key holiday periods and cyclical sine/cosine features
- Weather Variables: Weekly average temperature and sunshine duration
- ε_t: The model's error term for week t
```

---

## SUMMARY:
- **Fix 1:** Remove "and email" from promotional indicators
- **Fix 2:** Add the complete mathematical formula to show exactly what our model includes 