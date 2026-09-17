# Episode 01 Recording Script

## 00:00-00:30 — Opening

Today, we will use NBA data to learn two probability concepts. Our question is about LeBron James and his next season debut. How many field goals might he make in the 2026-27 opener? We will use his previous 21 season-debut performances as evidence. The goal is not to guarantee one exact prediction. It is to understand the PMF, PDF, and CDF clearly.

## 00:30-02:00 — Concepts

First, define a random variable called X. X is LeBron's field goals made in a season debut. Each historical season gives us one observed value of X. A probability distribution describes how likely different values are.

For a continuous variable, we often describe a PDF. PDF means probability density function, not point probability. Probability comes from the area under a PDF over an interval.

But field goals made can only be whole-number counts. Therefore, X is discrete rather than continuous. The correct discrete counterpart of a PDF is a PMF. A PMF assigns a probability to every possible count. All PMF probabilities must add up to one.

The CDF answers a different but related question. It gives the probability that X is at most k. Formally, F of k equals P of X less than or equal to k. We compute it by cumulatively adding PMF probabilities. So a CDF never decreases and eventually reaches one. The PMF means exactly; the CDF means at most.

## 02:00-02:30 — NBA Example

Now let us turn this idea into an NBA example. We use 21 completed seasons, from 2005-06 through 2025-26. The target is LeBron's season debut in 2026-27. For each season, we select his first regular-season appearance. That may differ from his team's scheduled opening night. This choice avoids treating a missed opener as zero field goals.

## 02:30-06:30 — Python Walkthrough

Let us open the Jupyter notebook and start with the imports. We import pandas, Matplotlib, Path, sleep, and nba_api. LeBron's NBA player identifier is stored as a constant. We also define the first year and number of seasons. The season-label function converts 2005 into 2005-06. A list comprehension generates all 21 season labels.

Next, we define a function that fetches each season debut. The rows list will collect one dictionary per season. We loop through the season labels in chronological order. PlayerGameLog requests LeBron's game log for one season. We explicitly restrict the request to regular-season games. The API response becomes a pandas DataFrame. An empty-result check prevents silent missing-season errors.

Next, we convert GAME_DATE into a real datetime column. We sort by date before selecting the first row. That first row represents LeBron's first appearance that season. We retain the season, date, matchup, and field goals made. Converting FGM to integer makes its discrete type explicit. Then we append the four values to our rows list. A short pause avoids sending requests too aggressively.

After the loop, the rows become a new DataFrame. We sort it chronologically and reset the row index. The result is cached as a CSV for reproducibility. The loader uses that cache when it already exists. That makes repeated notebook runs faster and more stable. We also validate that the table contains exactly 21 rows. Now inspect the table before calculating any probabilities. Each row is one season, and FGM is our observation.

We are ready to estimate the empirical distribution. The distribution function receives the FGM series. First, we create support from the minimum to maximum count. This includes every integer, even if one was never observed. Value counts tells us how often each FGM occurred. With normalize set to true, frequencies become probabilities. We sort the values and reindex them onto the full support. Unobserved counts receive probability zero, not missing values.

The PMF is complete, so the CDF is its cumulative sum. Each CDF value includes that count and everything below it. Next, we calculate three summaries for interpretation. The mean gives the historical average season-debut FGM. The PMF mode gives the most frequently observed count. The CDF estimates the probability of eight or fewer makes. These are empirical summaries, not a deterministic prediction.

Finally, Matplotlib creates two side-by-side charts. The left chart uses bars to display the PMF. The right chart uses steps to display the CDF. Read the left chart as probability of exactly each count. Read the right chart as probability up to each count.

## 06:30-07:00 — Closing

Here is the main takeaway from this short analysis. A PDF describes density for a continuous random variable. A PMF describes point probabilities for a discrete count. A CDF describes the probability of being at or below a value. Our baseline still ignores age, minutes, opponent, and injuries. In the next iteration, we can add those predictive features.
