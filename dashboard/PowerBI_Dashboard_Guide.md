# CareerInsight AI Power BI dashboard

Import these existing CSV files into Power BI Desktop:

1. `data/raw/Indian_Student_Placement_Dataset_2025.csv`
2. `data/raw/ai_job_salary_dataset_10k.csv`
3. `data/raw/DMA DATASET.csv`

Use the placement dataset for the main dashboard. Do not use `company_type` or `package_lpa` as predictive inputs because they are known only after an outcome. They remain suitable for historical descriptive visuals.

Recommended report pages:

## Placement overview

- Cards: total students, placed students, placement rate, average CGPA.
- Column chart: placement rate by degree and branch.
- Histogram: CGPA distribution split by placement status.
- Slicers: degree, branch, gender, backlogs.

## Student readiness

- Scatter chart: CGPA against aptitude score, colored by placement status.
- Clustered columns: average coding, communication, aptitude, projects, and internships by placement status.
- Matrix: branch by degree with student count and placement rate.

## Career and salary context

- Salary table: job role, country, experience level, salary USD, company type.
- Bar chart: median salary USD by job role.
- Slicers: country, experience level, company type.

Useful measures:

```DAX
Students = COUNTROWS('Indian_Student_Placement_Dataset_2025')

Placed Students = CALCULATE([Students], 'Indian_Student_Placement_Dataset_2025'[placed] = 1)

Placement Rate = DIVIDE([Placed Students], [Students])

Average CGPA = AVERAGE('Indian_Student_Placement_Dataset_2025'[cgpa])

Median Salary USD = MEDIAN('ai_job_salary_dataset_10k'[salary_usd])
```

Keep placement and salary pages separate: the supplied datasets do not contain a reliable student-to-job salary relationship. Label model outputs as estimates and show the deterministic data limitation in the report notes.
