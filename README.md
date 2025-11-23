## AI Planning Task1

You can run unline code and see result

```python
python class_assignment_final.py --input students.csv --output assignment.csv --mode ortools
```

# AI Planning Report
## Class Assignment Optimization System

---

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Thought Process](#thought-process)
3. [System Design](#system-design)
4. [Constraint Analysis](#constraint-analysis)
5. [Implementation Process](#implementation-process)
6. [Code Description](#code-description)
7. [Experimental Results](#experimental-results)

---

## Problem Definition

### Problem Situation
The problem involves assigning 200 students to 6 classes (4 classes of 33 students, 2 classes of 34 students) while simultaneously satisfying the following complex constraints:

1. **Enemy Separation**: Students who do not get along must not be assigned to the same class.
2. **Previous Classmate Dispersion**: Students who were in the same class last year should be dispersed so they do not concentrate in one class.
3. **Leadership Distribution**: At least one student with leadership qualities must be assigned to each class.
4. **Piano Student Distribution**: Students capable of playing the piano should be evenly distributed across classes.
5. **Grade Balance**: The average grade of each class should remain similar.
6. **Non-Attending (At-Risk) Student Distribution**: Students with a tendency for school refusal/non-attendance should be evenly distributed.
7. **Gender Balance**: The ratio of male and female students should be evenly distributed across classes.
8. **Athletic Ability Distribution**: Students who prefer sports should be evenly distributed.
9. **Club Activity Diversity**: Members of 10 different club types (Singing, Dance, Baseball, Art, Band, Soccer, Coding, Acting, Volunteering, Reading) must be evenly distributed so that every class has diverse club activities.

### Complexity of the Problem
- **NP-Hard Problem**: A combinatorial optimization problem where the number of variables increases exponentially.
- **Multiple Constraints**: 9 different constraints interact with each other.
- **Mixed Hard/Soft Constraints**: Some conditions must be met (Hard), while others are optimization goals (Soft).

---

## Thought Process

### Step 1: Problem Analysis
Initially thought of as a simple assignment problem, but discovered the following complexities:
- **Impossible Constraints**: It is impossible to completely separate all previous year classmates.
- **Unclear Criteria**: It was unclear whether "grade balance" meant the average should be the same or the total score sum should be the same.

### Step 2: Approach Selection
After considering several algorithms:
- **Greedy Algorithm**: Fast but does not guarantee an optimal solution.
- **Genetic Algorithm**: Complex and convergence is difficult to guarantee.
- **OR-Tools CP-SAT**: Specialized for handling constraints, guarantees an optimal solution.

**→ Selected OR-Tools CP-SAT**

### Step 3: Incremental Development Strategy
Implementing all constraints at once makes debugging difficult, so an incremental approach was adopted:
- Step 1: Basic constraints (assignment, headcount, enemy relations)
- Step 2: Add leadership distribution
- Step 3: Add piano student distribution
- ...
- Step 11: Complete all constraints

### Step 4: Constraint Classification
Classified constraints for effective implementation:
- **Hard Constraints**: Must be satisfied.
- **Soft Constraints**: Optimized via an objective function.

---

## System Design

### Architecture Overview
```
CSV Data → Data Processing → OR-Tools CP-SAT → Optimization → Results
    ↓            ↓                 ↓               ↓            ↓
Student Info  Norm/Mapping   Constraint Gen.   Obj. Function   Assignment
```

### Core Components
1. **Data Preprocessing Module**: CSV parsing, data normalization.
2. **Constraint Generator**: Converts each requirement into OR-Tools constraints.
3. **Optimization Engine**: Searches for the optimal solution using the CP-SAT solver.
4. **Result Analyzer**: Evaluates and visualizes the quality of the assignment results.

---

## Constraint Analysis

### Hard Constraints
**1. Basic Assignment Constraints**
```python
# Each student is assigned to exactly one class
for i in range(n):
    model.Add(sum(x[(i,c)] for c in range(k)) == 1)

# Maintain exact number of students per class
for c in range(k):
    model.Add(sum(x[(i,c)] for i in range(n)) == class_sizes[c])
```

**2. Enemy Separation**
```python
for i in range(n):
    for j in enemies_list[i]:
        for c in range(k):
            model.Add(x[(i,c)] + x[(j,c)] <= 1)  # Ban assignment to same class
```

**3. Leadership Student Distribution**
```python
# Assign at least 1 leadership student per class
for c in range(k):
    model.Add(sum(x[(i,c)] for i in leadership_students) >= 1)
```

**4. Piano/Non-Attending/Athletic Student Distribution**
```python
# Evenly distribute special attribute students (3-4 per class)
for attribute_group in [piano_students, at_risk_students, athletic_students]:
    total = len(attribute_group)
    floor_count = total // k
    ceil_count = math.ceil(total / k)
    for c in range(k):
        model.Add(sum(x[(i,c)] for i in attribute_group) >= floor_count)
        model.Add(sum(x[(i,c)] for i in attribute_group) <= ceil_count)
```

**5. Gender Balance**
```python
# Assign male students evenly (23-24 per class) - females balance automatically
male_total = len(male_students)
male_floor = male_total // k
male_ceil = math.ceil(male_total / k)
for c in range(k):
    model.Add(sum(x[(i,c)] for i in male_students) >= male_floor)
    model.Add(sum(x[(i,c)] for i in male_students) <= male_ceil)
```

**6. Club Diversity Assurance**
```python
# Individual even distribution for each club type (10 types)
for club_name, club_members in club_groups.items():
    club_total = len(club_members)
    club_floor = club_total // k
    club_ceil = math.ceil(club_total / k)
    for c in range(k):
        model.Add(sum(x[(i,c)] for i in club_members) >= club_floor)
        model.Add(sum(x[(i,c)] for i in club_members) <= club_ceil)
```

### Soft Constraints
**1. Previous Class Dispersion**
```python
# Prevent too many students from the same previous class gathering in one class
max_per_class = math.ceil(len(students) / k) + 1
over_var = model.NewIntVar(0, len(students), f"prev_over_{prev_class}_{c}")
model.Add(over_var >= sum(x[(i,c)] for i in students) - max_per_class)
```

**2. Grade Balance**
```python
# Minimize grade deviation between classes
max_deviation = model.NewIntVar(0, total_grade, 'max_deviation')
for c in range(k):
    model.Add(class_grade_sum[c] - target_grade_per_class <= max_deviation)
    model.Add(target_grade_per_class - class_grade_sum[c] <= max_deviation)
```

---

## Implementation Process

### Phase 1: Data Preprocessing
**Data Structure Analysis**:
- Student Info: Complete dataset of 200 students.
- Enemy Relations: 20 specific relationship entries.
- Previous Class: Origin info from 6 classes (a~f).
- Attribute Info: Leadership, Piano, Non-attending, Gender, Athletic, Club, etc., fully available.

**Implemented Utility Functions**:
```python
def find_col(cols, possible_names):
    # Universal function to handle various column name patterns
    for name in possible_names:
        if name in cols:
            return name
    for col in cols:
        if col.lower() in [name.lower() for name in possible_names]:
            return col

def normalize_bool(x):
    # Normalize boolean values (0/1 conversion)
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    return 1 if s in ('1','yes','y','true','exist','leader') else 0
```

### Phase 2: Incremental Constraint Addition
Created independent files for each step to facilitate debugging:
- `class_assignment_step3.py`: Basic constraints
- `class_assignment_step4.py`: Added leadership
- ...
- `class_assignment_final.py`: Integrated all constraints

### Phase 3: Performance Optimization
**Problem**: Attempted complete separation of previous classmates.
- '24 Class Sizes: a(36), b(35), c(35), d(36), e(26), f(32).
- To completely separate all previous classmates, **36 classes are needed** (based on the largest origin class).
- With only 6 current classes, this is mathematically impossible, resulting in an INFEASIBLE status.

**Solution**: Changed from Hard Constraint to Soft Constraint.
```python
# Before: Separate all pairs of previous classmates (Hard Constraint - INFEASIBLE)
for i, j in all_previous_pairs:
    for c in range(k):
        model.Add(x[(i,c)] + x[(j,c)] <= 1)

# After: Disperse so not too many from same origin are in one class (Soft Constraint)
max_per_class = math.ceil(len(students) / k) + 1  # Max 6-7 per class
over_var = model.NewIntVar(0, len(students), f"prev_over_{prev_class}_{c}")
model.Add(over_var >= sum(x[(i,c)] for i in students) - max_per_class)
model.Minimize(sum(all_over_vars))  # Add to objective function to optimize
```

### Data Structure
```python
# Student Information Structure
Student = {
    'id': '202501',
    'name': 'Andy',
    'gender': 'boy',
    'score': 90,
    'previous_class': 'a',
    'club': 'Singing',
    'enemies': ['202502', '202503'],
    'leadership': True,
    'piano': False,
    'at_risk': False,
    'athletic': False
}
```

---

## Code Description

### Core Function Analysis

**1. Main Execution Function**
```python
def run_ortools_final(students_df, class_sizes):
    """
    Final Class Assignment Optimization using OR-Tools CP-SAT
    - Applies all 9 constraints
    - Combines Hard and Soft constraints
    - Searches for OPTIMAL solution
    """
```

**2. Data Preprocessing Functions**
```python
def find_col(cols, possible_names):
    """Matches various column name patterns"""
    
def normalize_bool(x):
    """Normalizes various boolean expressions"""
    
def build_previous_classmates(df, prev_class_col):
    """Builds previous year class relationships"""
```

**3. Constraint Generation Functions**
- `add_enemy_constraints()`: Enemy separation
- `add_previous_class_distribution()`: Previous class dispersion
- `add_attribute_balance()`: Attribute-based even distribution
- `add_club_diversity()`: Club diversity assurance
- `add_grade_balance()`: Grade balance optimization

### Optimization Strategy
1. **Variable Definition**: 200×6 binary variable matrix.
2. **Constraint Priority**: Hard Constraints → Soft Constraints.
3. **Objective Function**: Minimize grade deviation + Optimize previous class dispersion.
4. **Solver Settings**: Default CP-SAT settings are sufficient.

---

## Experimental Results

### Final Performance Metrics
- **Solver Status**: OPTIMAL (Optimal solution achieved)
- **Execution Time**: Approx. 30-60 seconds
- **Variable Count**: 1,200 (200 students × 6 classes)
- **Constraint Count**: Approx. 300

### Satisfaction by Constraint

| Constraint | Goal | Result | Achievement |
|:---|:---|:---|:---|
| Enemy Separation | 100% | 100% | Perfect |
| Prev. Class Dispersion | Max 6-7 per class | All classes satisfied | Perfect |
| Leadership Dist. | Min 1 per class | 1-7 distributed | Perfect |
| Piano Student Dist. | 3-4 per class | 3-4 per class | Perfect |
| Grade Balance | Min deviation | Std Dev 0.0 | Perfect |
| Non-Attending Dist. | 3-4 per class | 3-4 per class | Perfect |
| Gender Balance | 23-24 per class | 23-24 per class | Perfect |
| Athletic Ability Dist. | 3-4 per class | 3-4 per class | Perfect |
| Club Diversity | Even dist. per club | All classes have 10 types | Perfect |

### Detailed Result Analysis

**1. Grade Balance (Perfectly Achieved)**
```
Class 0: 33 students, Total Score 2631 (Avg 79.7)
Class 1: 33 students, Total Score 2631 (Avg 79.7)  
Class 2: 33 students, Total Score 2631 (Avg 79.7)
Class 3: 33 students, Total Score 2631 (Avg 79.7)
Class 4: 34 students, Total Score 2631 (Avg 77.4)
Class 5: 34 students, Total Score 2631 (Avg 77.4)
Score Total Standard Deviation: 0.0 (Perfect Balance)
```

**2. Club Diversity (10 Club Types in Every Class)**
```
Acting Club (26): Cls0:4, Cls1:4, Cls2:4, Cls3:4, Cls4:5, Cls5:5
Art Club (25): Cls0:4, Cls1:4, Cls2:5, Cls3:4, Cls4:4, Cls5:4
Band Club (21): Cls0:4, Cls1:4, Cls2:3, Cls3:3, Cls4:3, Cls5:4
Soccer Club (21): Cls0:4, Cls1:3, Cls2:4, Cls3:3, Cls4:4, Cls5:3
Baseball Club (20): Cls0:3, Cls1:4, Cls2:3, Cls3:4, Cls4:3, Cls5:3
```

**3. Previous Class Dispersion**
```
Prev Class a (36): Distributed max 7 or less per class
Prev Class b (35): Distributed max 7 or less per class
Prev Class c (35): Distributed max 7 or less per class
...
```

### Performance Improvement Process
| Version | Issue | Solution | Result |
|:---|:---|:---|:---|
| Step 3-8 | Complex constraints -> FEASIBLE | Incremental addition | OPTIMAL |
| Initial Final | Prev. Class Separation INFEASIBLE | Hard → Soft Constraint | FEASIBLE |
| Final Final | Grade balance optimization | Balanced by Total Score | OPTIMAL, Dev 0.0 |
