#!/usr/bin/env python3
"""
class_assignment_step10.py - 전년도 클래스메이트 분리 추가

Step 10: 전년도 클래스메이트 분리 추가
1. 각 학생은 정확히 하나의 학급에 배정 
2. 각 학급의 정확한 인원수 유지 (33명×4클래스, 34명×2클래스)
3. 적대관계인 학생들은 같은 학급에 배정하지 않음 (제약조건 1-A)
4. 전년도 같은 클래스였던 학생들은 가능한 분리 (제약조건 1-B)
5. 리더십을 가진 학생이 각 학급에 최소 1명씩 배정 (제약조건 2)
6. 피아노 연주 가능한 학생을 균등하게 분배 (제약조건 3)
7. 성적·학력을 균등하게 분배 (제약조건 4)
8. 비등교자가 치우치지 않도록 균등하게 분배 (제약조건 5)
9. 남녀 비율을 균등하게 분배 (제약조건 6)
10. 운동 능력(발이 빠른 아이)을 균등하게 분배 (제약조건 7)

Usage:
    python class_assignment_step10.py --input students.csv --output assignment.csv --mode ortools
"""

import argparse
import pandas as pd
import numpy as np
import math

def find_col(cols, possible_names):
    for name in possible_names:
        if name in cols:
            return name
    for col in cols:
        low = col.lower()
        for name in possible_names:
            if low == name.lower():
                return col
    return None

def parse_list_field(cell):
    if pd.isna(cell) or str(cell).strip()=='':
        return []
    s = str(cell)
    sep = ',' if ',' in s else ';' if ';' in s else '/'
    parsed = []
    for p in s.split(sep):
        p = p.strip()
        if p != '':
            try:
                if '.' in p:
                    p = str(int(float(p)))
            except:
                pass
            parsed.append(p)
    return parsed

def normalize_bool(x):
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in ('1','yes','y','true','t','o','예','있음','있다','leader','리더'):
        return 1
    try:
        f = float(s)
        return 1 if f>0 else 0
    except:
        if any(k in s for k in ['yes','true','leader','리더','피아노','piano','비등교','drop','absent']):
            return 1
    return 0

def normalize_gender_male(x):
    """성별에서 남성 여부를 판단하는 전용 함수"""
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in ('male','m','boy','남','남자','남성'):
        return 1
    if any(k in s for k in ['male','boy','남자','남성']):
        return 1
    return 0

def resolve_refs(raw_list, id_to_idx, n):
    res = []
    for token in raw_list:
        t = token.strip()
        if t == '': continue
        if t in id_to_idx:
            res.append(id_to_idx[t])
        else:
            try:
                ii = int(t)
                if 0 <= ii < n:
                    res.append(ii)
            except:
                print(f"⚠️ 존재하지 않는 참조 무시: {t}")
    return res

def run_ortools_step10(df, out_path, class_sizes):
    try:
        from ortools.sat.python import cp_model
    except Exception as e:
        raise RuntimeError("ortools not installed. Install with `pip install ortools`") from e

    n = len(df)
    k = len(class_sizes)

    model = cp_model.CpModel()

    # Variables x[i,c] binary
    x = {}
    for i in range(n):
        for c in range(k):
            x[(i,c)] = model.NewBoolVar(f"x_{i}_{c}")

    print(f"Created {n * k} variables")

    # 제약조건 1: 각 학생은 정확히 하나의 학급에 배정
    for i in range(n):
        model.Add(sum(x[(i,c)] for c in range(k)) == 1)
    print("Added student assignment constraints")

    # 제약조건 2: 각 학급의 정확한 인원수 유지
    for c in range(k):
        model.Add(sum(x[(i,c)] for i in range(n)) == class_sizes[c])
    print(f"Added class size constraints: {class_sizes}")

    # 제약조건 3: 적대관계 분리 (1-A)
    enemy_constraints = 0
    for i in range(n):
        enemies_list = df.at[i,'enemies_idx']
        if isinstance(enemies_list, list):
            for j in enemies_list:
                if j>=0 and j<n and j!=i:
                    for c in range(k):
                        model.Add(x[(i,c)] + x[(j,c)] <= 1)
                    enemy_constraints += 1
    print(f"Added {enemy_constraints} enemy separation constraints")

    # 제약조건 4: 전년도 클래스메이트 분리 (1-B) - 소프트 제약으로 구현
    previous_constraints = 0
    violation_vars = []
    for i in range(n):
        previous_list = df.at[i,'previous_idx']
        if isinstance(previous_list, list):
            for j in previous_list:
                if j>=0 and j<n and j!=i:
                    # 소프트 제약: 같은 클래스에 배정되면 penalty
                    for c in range(k):
                        violation_var = model.NewBoolVar(f"prev_violation_{i}_{j}_{c}")
                        model.Add(x[(i,c)] + x[(j,c)] - 1 <= violation_var)
                        violation_vars.append(violation_var)
                    previous_constraints += 1
    print(f"Added {previous_constraints} previous classmate separation (soft) constraints")

    # 제약조건 5: 리더십 분배 (2번) - 각 학급에 최소 1명씩
    leader_idxs = [i for i in range(n) if df.at[i,'is_leader']==1]
    if leader_idxs:
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in leader_idxs) >= 1)
        print(f"Added leadership constraints: {len(leader_idxs)} leaders, min 1 per class")

    # 제약조건 6: 피아노 학생 균등 분배 (3번)
    piano_idxs = [i for i in range(n) if df.at[i,'is_piano']==1]
    if piano_idxs:
        p_total = len(piano_idxs)
        p_floor = p_total // k
        p_ceil = math.ceil(p_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in piano_idxs) >= p_floor)
            model.Add(sum(x[(i,c)] for i in piano_idxs) <= p_ceil)
        print(f"Added piano balance constraints: {p_total} piano students, {p_floor}-{p_ceil} per class")

    # 제약조건 7: 비등교자 균등 분배 (5번)
    atrisk_idxs = [i for i in range(n) if df.at[i,'is_at_risk']==1]
    if atrisk_idxs:
        a_total = len(atrisk_idxs)
        a_floor = a_total // k
        a_ceil = math.ceil(a_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in atrisk_idxs) >= a_floor)
            model.Add(sum(x[(i,c)] for i in atrisk_idxs) <= a_ceil)
        print(f"Added at-risk balance constraints: {a_total} at-risk students, {a_floor}-{a_ceil} per class")

    # 제약조건 8: 남녀 비율 균등 분배 (6번)
    male_idxs = [i for i in range(n) if df.at[i,'gender_m']==1]
    if male_idxs:
        m_total = len(male_idxs)
        m_floor = m_total // k
        m_ceil = math.ceil(m_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in male_idxs) >= m_floor)
            model.Add(sum(x[(i,c)] for i in male_idxs) <= m_ceil)
        print(f"Added gender balance constraints: {m_total} male students, {m_floor}-{m_ceil} per class")

    # 제약조건 9: 운동 능력 균등 분배 (7번)
    athletic_idxs = [i for i in range(n) if df.at[i,'is_athletic']==1]
    if athletic_idxs:
        ath_total = len(athletic_idxs)
        ath_floor = ath_total // k
        ath_ceil = math.ceil(ath_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in athletic_idxs) >= ath_floor)
            model.Add(sum(x[(i,c)] for i in athletic_idxs) <= ath_ceil)
        print(f"Added athletic balance constraints: {ath_total} athletic students, {ath_floor}-{ath_ceil} per class")

    # 제약조건 10: 성적 균형 분배 (4번) - 목적함수의 일부로 구현
    scale = 1
    grades_int = [int(round(v * scale)) for v in df['grade_val'].tolist()]
    total_grade = sum(grades_int)
    target_grade_per_class = total_grade // k
    
    print(f"Grade balancing: total={total_grade}, target per class={target_grade_per_class}")
    
    # 각 학급의 성적 합
    class_grade_sum = []
    for c in range(k):
        grade_sum = sum(x[(i,c)] * grades_int[i] for i in range(n))
        class_grade_sum.append(grade_sum)
    
    # 복합 목적함수: 성적 분산 최소화 + 전년도 클래스메이트 분리 최대화
    max_deviation = model.NewIntVar(0, total_grade, 'max_deviation')
    for c in range(k):
        model.Add(class_grade_sum[c] - target_grade_per_class <= max_deviation)
        model.Add(target_grade_per_class - class_grade_sum[c] <= max_deviation)
    
    # 목적함수: 성적 분산 최소화 (주 목표) + 전년도 분리 위반 최소화 (부 목표)
    total_violations = sum(violation_vars)
    model.Minimize(max_deviation * 1000 + total_violations)
    print("Added combined objective: grade balance (primary) + previous classmate separation (secondary)")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180  # 더 복잡해져서 시간 연장
    print("Solving CP-SAT...")
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"✅ SUCCESS: {solver.StatusName(status)}")
        
        # 결과 저장
        assign = []
        for i in range(n):
            for c in range(k):
                if solver.Value(x[(i,c)]) == 1:
                    assign.append((i, c))
                    break
        
        out_df = df.copy()
        out_df['assigned_class'] = [c for (_, c) in sorted(assign, key=lambda x:x[0])]
        out_df.to_csv(out_path, index=False)
        print(f"Saved assignment to {out_path}")
        
        # 결과 분석
        print("\n📊 결과 분석:")
        grade_sums = []
        total_prev_violations = 0
        
        for c in range(k):
            members = out_df[out_df['assigned_class']==c]
            leaders = members['is_leader'].sum()
            pianos = members['is_piano'].sum()
            atrisks = members['is_at_risk'].sum()
            males = members['gender_m'].sum()
            females = len(members) - males
            athletics = members['is_athletic'].sum()
            grade_avg = members['grade_val'].mean()
            grade_sum = members['grade_val'].sum()
            grade_sums.append(grade_sum)
            print(f"학급 {c}: {len(members)}명, 리더 {leaders}명, 피아노 {pianos}명, 비등교 {atrisks}명, 남 {males}명, 여 {females}명, 운동 {athletics}명, 평균성적 {grade_avg:.1f}, 총점 {grade_sum:.0f}")
        
        # 전년도 클래스메이트 분리 효과 측정
        if violation_vars:
            total_prev_violations = sum(solver.Value(v) for v in violation_vars)
            prev_separation_rate = 1 - (total_prev_violations / len(violation_vars)) if violation_vars else 1
            print(f"\n전년도 클래스메이트 분리: {total_prev_violations}/{len(violation_vars)} 위반 (분리율: {prev_separation_rate:.1%})")
        
        grade_std = np.std(grade_sums)
        print(f"성적 균형: 총점 표준편차 = {grade_std:.1f}")
        
        return True
    else:
        print(f"❌ FAILED: {solver.StatusName(status)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Class assignment - Step 10 (Previous classmate separation)")
    parser.add_argument('--input', required=True, help='input CSV file')
    parser.add_argument('--output', required=True, help='output CSV file')
    parser.add_argument('--mode', choices=['ortools'], default='ortools', help='solver mode')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]

    # 컬럼 감지
    cols = df.columns.tolist()
    id_col = find_col(cols, ['id','student_id','학번','ID'])
    enemies_col = find_col(cols, ['enemy','enemies','나쁜관계','안좋다','사이','hate'])
    previous_col = find_col(cols, ['previous','prev','전년도','작년','같은반','classmate'])
    leader_col = find_col(cols, ['leader','leadership','Leadership','리더','리더십'])
    piano_col = find_col(cols, ['piano','Piano','피아노'])
    grade_col = find_col(cols, ['score','grade','성적','학력','점수'])
    at_risk_col = find_col(cols, ['비등교','absent','non_attend','비등교성향','등교거부','drop'])
    gender_col = find_col(cols, ['gender','sex','성별'])
    athletic_col = find_col(cols, ['운동선호','athletic','sports','운동','체육'])

    print("=== Step 10: 전년도 클래스메이트 분리 추가 ===")
    print(f"감지된 컬럼들:")
    print(f"  ID: {id_col}")
    print(f"  적대관계: {enemies_col}")
    print(f"  전년도: {previous_col}")
    print(f"  리더십: {leader_col}")
    print(f"  피아노: {piano_col}")
    print(f"  성적: {grade_col}")
    print(f"  비등교: {at_risk_col}")
    print(f"  성별: {gender_col}")
    print(f"  운동: {athletic_col}")

    # 데이터 처리
    n = len(df)
    
    # ID 매핑
    id_to_idx = {}
    for idx, row in df.iterrows():
        sid = str(row[id_col]).strip() if id_col in df.columns else str(idx)
        id_to_idx[sid] = idx

    # 적대관계 처리
    if enemies_col and enemies_col in df.columns:
        enemies_raw = df[enemies_col].apply(parse_list_field)
        enemies_idx = enemies_raw.apply(lambda x: resolve_refs(x, id_to_idx, n))
        df['enemies_idx'] = enemies_idx
        total_enemy_pairs = sum(len(enemies) for enemies in enemies_idx)
    else:
        df['enemies_idx'] = [[] for _ in range(n)]
        total_enemy_pairs = 0

    # 전년도 클래스메이트 처리
    if previous_col and previous_col in df.columns:
        previous_raw = df[previous_col].apply(parse_list_field)
        previous_idx = previous_raw.apply(lambda x: resolve_refs(x, id_to_idx, n))
        df['previous_idx'] = previous_idx
        total_previous_pairs = sum(len(prev) for prev in previous_idx)
    else:
        df['previous_idx'] = [[] for _ in range(n)]
        total_previous_pairs = 0

    # 리더십 처리
    if leader_col and leader_col in df.columns:
        df['is_leader'] = df[leader_col].apply(normalize_bool)
        total_leaders = df['is_leader'].sum()
    else:
        df['is_leader'] = 0
        total_leaders = 0

    # 피아노 처리
    if piano_col and piano_col in df.columns:
        df['is_piano'] = df[piano_col].apply(normalize_bool)
        total_pianos = df['is_piano'].sum()
    else:
        df['is_piano'] = 0
        total_pianos = 0

    # 비등교 처리
    if at_risk_col and at_risk_col in df.columns:
        df['is_at_risk'] = df[at_risk_col].apply(normalize_bool)
        total_atrisks = df['is_at_risk'].sum()
    else:
        df['is_at_risk'] = 0
        total_atrisks = 0

    # 성별 처리
    if gender_col and gender_col in df.columns:
        df['gender_m'] = df[gender_col].apply(normalize_gender_male)
        total_males = df['gender_m'].sum()
        total_females = n - total_males
    else:
        df['gender_m'] = 0
        total_males = 0
        total_females = n

    # 운동 능력 처리
    if athletic_col and athletic_col in df.columns:
        df['is_athletic'] = df[athletic_col].apply(normalize_bool)
        total_athletics = df['is_athletic'].sum()
    else:
        df['is_athletic'] = 0
        total_athletics = 0

    # 성적 처리
    if grade_col and grade_col in df.columns:
        def to_num(x):
            try:
                return float(x)
            except:
                import re
                m = re.search(r'\d+(\.\d+)?', str(x))
                return float(m.group(0)) if m else 0.0
        df['grade_val'] = df[grade_col].apply(to_num)
        avg_grade = df['grade_val'].mean()
    else:
        df['grade_val'] = 0.0
        avg_grade = 0.0

    # 학급 크기: 33명×4클래스, 34명×2클래스
    class_sizes = [33, 33, 33, 33, 34, 34]

    print(f"\n📋 데이터 요약:")
    print(f"  총 학생 수: {n}")
    print(f"  학급 구성: {class_sizes}")
    print(f"  적대관계: {total_enemy_pairs}건")
    print(f"  전년도 클래스메이트: {total_previous_pairs}건")
    print(f"  리더십 학생: {total_leaders}명")
    print(f"  피아노 학생: {total_pianos}명")
    print(f"  비등교 학생: {total_atrisks}명")
    print(f"  남학생: {total_males}명, 여학생: {total_females}명")
    print(f"  운동 선호 학생: {total_athletics}명")
    print(f"  평균 성적: {avg_grade:.1f}")

    # OR-Tools 실행
    success = run_ortools_step10(df, args.output, class_sizes)
    if not success:
        print("\n💡 실패 원인을 분석해보세요.")

if __name__ == '__main__':
    main()