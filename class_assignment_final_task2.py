#!/usr/bin/env python3
"""
class_assignment_final_task2.py - Modified for task2.csv

This file is a copy of class_assignment_final.py with small enhancements:
 - Enforce sibling groups (connected components) so all siblings are in the same class
 - Make transfer-preference weight configurable (--transfer-weight)
 - Make submission bonus configurable (--submission-bonus)
 - Default input to task2.csv and output to assignment_task2.csv for convenience

Usage:
    python class_assignment_final_task2.py --input task2.csv --output assignment_task2.csv
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
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in ('male','m','boy','남','남자','남성'):
        return 1
    if any(k in s for k in ['male','boy','남자','남성']):
        return 1
    return 0


def normalize_club(x):
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in ('', 'nan', 'none', 'null'):
        return 0
    if len(s) > 0:
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


def build_previous_classmates(df, previous_class_col):
    n = len(df)
    previous_classmates = [[] for _ in range(n)]
    if not previous_class_col or previous_class_col not in df.columns:
        print("⚠️ 전년도 학급 정보가 없습니다.")
        return previous_classmates, 0
    class_groups = {}
    for idx, row in df.iterrows():
        prev_class = str(row[previous_class_col]).strip().lower()
        if prev_class and prev_class != 'nan':
            if prev_class not in class_groups:
                class_groups[prev_class] = []
            class_groups[prev_class].append(idx)
    total_pairs = 0
    for class_name, students in class_groups.items():
        if len(students) > 1:
            print(f"전년도 {class_name}반: {len(students)}명")
            for i in range(len(students)):
                for j in range(i+1, len(students)):
                    student_i = students[i]
                    student_j = students[j]
                    previous_classmates[student_i].append(student_j)
                    previous_classmates[student_j].append(student_i)
                    total_pairs += 1
    return previous_classmates, total_pairs


def build_sibling_groups(siblings_idx):
    # siblings_idx: Series or list of lists mapping i -> list of sibling indices
    n = len(siblings_idx)
    visited = [False]*n
    groups = []

    # build adjacency
    adj = [[] for _ in range(n)]
    for i, lst in enumerate(siblings_idx):
        if isinstance(lst, list):
            for j in lst:
                if 0 <= j < n and j != i:
                    adj[i].append(j)
                    adj[j].append(i)

    for i in range(n):
        if not visited[i]:
            stack = [i]
            comp = []
            while stack:
                u = stack.pop()
                if visited[u]:
                    continue
                visited[u] = True
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        stack.append(v)
            groups.append(comp)

    return groups


def run_ortools_final(df, out_path, class_sizes, club_col=None, previous_class_col=None, transfer_weight=5):
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

    # 제약조건 2: 각 학급의 정확한 인원수 유지
    for c in range(k):
        model.Add(sum(x[(i,c)] for i in range(n)) == class_sizes[c])

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

    # 제약조건 4: 전년도 클래스메이트 분산 (soft)
    violation_vars = []
    if previous_class_col and previous_class_col in df.columns:
        prev_class_groups = {}
        for idx, row in df.iterrows():
            prev_class = str(row[previous_class_col]).strip().lower()
            if prev_class and prev_class not in ('', 'nan', 'none', 'null'):
                if prev_class not in prev_class_groups:
                    prev_class_groups[prev_class] = []
                prev_class_groups[prev_class].append(idx)
        for prev_class, students in prev_class_groups.items():
            if len(students) > k:
                max_per_class = math.ceil(len(students) / k) + 1
                for c in range(k):
                    over_var = model.NewIntVar(0, len(students), f"prev_over_{prev_class}_{c}")
                    model.Add(over_var >= sum(x[(i,c)] for i in students) - max_per_class)
                    model.Add(over_var >= 0)
                    violation_vars.append(over_var)

    # 제약조건 5: 리더십 분배
    leader_idxs = [i for i in range(n) if df.at[i,'is_leader']==1]
    if leader_idxs:
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in leader_idxs) >= 1)

    # 피아노, 비등교, 성별, 운동, 클럽 등은 원본과 동일하게 적용
    piano_idxs = [i for i in range(n) if df.at[i,'is_piano']==1]
    if piano_idxs:
        p_total = len(piano_idxs)
        p_floor = p_total // k
        p_ceil = math.ceil(p_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in piano_idxs) >= p_floor)
            model.Add(sum(x[(i,c)] for i in piano_idxs) <= p_ceil)

    atrisk_idxs = [i for i in range(n) if df.at[i,'is_at_risk']==1]
    if atrisk_idxs:
        a_total = len(atrisk_idxs)
        a_floor = a_total // k
        a_ceil = math.ceil(a_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in atrisk_idxs) >= a_floor)
            model.Add(sum(x[(i,c)] for i in atrisk_idxs) <= a_ceil)

    male_idxs = [i for i in range(n) if df.at[i,'gender_m']==1]
    if male_idxs:
        m_total = len(male_idxs)
        m_floor = m_total // k
        m_ceil = math.ceil(m_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in male_idxs) >= m_floor)
            model.Add(sum(x[(i,c)] for i in male_idxs) <= m_ceil)

    athletic_idxs = [i for i in range(n) if df.at[i,'is_athletic']==1]
    if athletic_idxs:
        ath_total = len(athletic_idxs)
        ath_floor = ath_total // k
        ath_ceil = math.ceil(ath_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in athletic_idxs) >= ath_floor)
            model.Add(sum(x[(i,c)] for i in athletic_idxs) <= ath_ceil)

    if club_col and club_col in df.columns:
        club_groups = {}
        for idx, row in df.iterrows():
            club_name = str(row[club_col]).strip().lower()
            if club_name and club_name not in ('', 'nan', 'none', 'null'):
                if club_name not in club_groups:
                    club_groups[club_name] = []
                club_groups[club_name].append(idx)
        for club_name, club_members in club_groups.items():
            if len(club_members) >= k:
                club_total = len(club_members)
                club_floor = club_total // k
                club_ceil = math.ceil(club_total / k)
                for c in range(k):
                    model.Add(sum(x[(i,c)] for i in club_members) >= club_floor)
                    model.Add(sum(x[(i,c)] for i in club_members) <= club_ceil)
            else:
                for c in range(k):
                    model.Add(sum(x[(i,c)] for i in club_members) <= 1)

    # 형제자매 그룹: 데이터프레임에 'sibling_groups'가 있으면 그룹 단위로 같은 반 enforced
    if 'sibling_groups' in df.columns:
        # convert to plain Python list to avoid pandas Series truth-value ambiguity
        sibling_groups = df['sibling_groups'].tolist()
    else:
        sibling_groups = []
    sibling_links = 0
    for group in sibling_groups:
        if isinstance(group, list) and len(group) > 1:
            rep = group[0]
            for i in group[1:]:
                for c in range(k):
                    model.Add(x[(rep,c)] == x[(i,c)])
                    sibling_links += 1

    # 성적 균형 목적
    scale = 1
    grades_int = [int(round(v * scale)) for v in df['grade_val'].tolist()]
    total_grade = sum(grades_int)
    target_grade_per_class = total_grade // k
    class_grade_sum = []
    for c in range(k):
        grade_sum = sum(x[(i,c)] * grades_int[i] for i in range(n))
        class_grade_sum.append(grade_sum)

    max_deviation = model.NewIntVar(0, total_grade, 'max_deviation')
    for c in range(k):
        model.Add(class_grade_sum[c] - target_grade_per_class <= max_deviation)
        model.Add(target_grade_per_class - class_grade_sum[c] <= max_deviation)

    total_violations = sum(violation_vars) if violation_vars else 0

    # 전학생 우선 배정 보상
    transfer_idxs = [i for i in range(n) if 'is_transfer' in df.columns and df.at[i,'is_transfer']==1]
    transfer_score_terms = []
    if transfer_idxs and leader_idxs:
        leader_present = []
        for c in range(k):
            lp = model.NewBoolVar(f"leader_present_{c}")
            model.Add(sum(x[(i,c)] for i in leader_idxs) >= lp)
            model.Add(sum(x[(i,c)] for i in leader_idxs) <= lp * len(leader_idxs))
            leader_present.append(lp)

        for t in transfer_idxs:
            for c in range(k):
                z_tc = model.NewBoolVar(f"transfer_{t}_leader_in_{c}")
                model.Add(z_tc <= x[(t,c)])
                model.Add(z_tc <= leader_present[c])
                model.AddBoolAnd([x[(t,c)], leader_present[c]]).OnlyEnforceIf(z_tc)
                model.AddBoolOr([x[(t,c)].Not(), leader_present[c].Not()]).OnlyEnforceIf(z_tc.Not())
                transfer_score_terms.append(z_tc)

    if transfer_score_terms:
        model.Minimize(max_deviation * 10000 + total_violations - transfer_weight * sum(transfer_score_terms))
    else:
        model.Minimize(max_deviation * 10000 + total_violations)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600
    print("Solving CP-SAT...")
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"✅ SUCCESS: {solver.StatusName(status)}")
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

        # 상세 결과 분석 및 조건별 성공 여부 출력
        print("\n🎯 최종 결과 분석:")
        grade_sums = []
        leader_ok = True
        sibling_ok = True
        gender_ok = True
        piano_ok = True
        atrisk_ok = True
        athletic_ok = True
        club_ok = True
        class_size_ok = True
        # Sibling check: for each sibling group, all assigned_class must be identical
        for group in sibling_groups:
            if len(group) > 1:
                assigned = set(out_df.loc[group, 'assigned_class'])
                if len(assigned) > 1:
                    sibling_ok = False
                    break

        for c in range(k):
            members = out_df[out_df['assigned_class']==c]
            leaders = members['is_leader'].sum()
            pianos = members['is_piano'].sum()
            atrisks = members['is_at_risk'].sum()
            males = members['gender_m'].sum()
            females = len(members) - males
            athletics = members['is_athletic'].sum()
            clubs = members['is_club'].sum()
            grade_avg = members['grade_val'].mean()
            grade_sum = members['grade_val'].sum()
            grade_sums.append(grade_sum)
            # Leader check
            if leaders < 1:
                leader_ok = False
            # Gender balance check (allow ±5)
            if abs(males - females) > 5:
                gender_ok = False
            # Piano check (allow ±2)
            piano_total = out_df['is_piano'].sum()
            piano_per_class = piano_total // k
            if not (piano_per_class-2 <= pianos <= piano_per_class+2):
                piano_ok = False
            # At-risk check (allow ±2)
            atrisk_total = out_df['is_at_risk'].sum()
            atrisk_per_class = atrisk_total // k
            if not (atrisk_per_class-2 <= atrisks <= atrisk_per_class+2):
                atrisk_ok = False
            # Athletic check (allow ±2)
            athletic_total = out_df['is_athletic'].sum()
            athletic_per_class = athletic_total // k
            if not (athletic_per_class-2 <= athletics <= athletic_per_class+2):
                athletic_ok = False
            # Club check (allow ±2)
            club_total = out_df['is_club'].sum()
            club_per_class = club_total // k
            if not (club_per_class-2 <= clubs <= club_per_class+2):
                club_ok = False
            # Class size check
            if len(members) != class_sizes[c]:
                class_size_ok = False
            print(f"학급 {c}: {len(members)}명, 리더 {leaders}명, 피아노 {pianos}명, 비등교 {atrisks}명, 남 {males}명, 여 {females}명, 운동 {athletics}명, 클럽 {clubs}명, 평균성적 {grade_avg:.1f}, 총점 {grade_sum:.0f}")

        grade_std = np.std(grade_sums)
        print(f"\n성적 균형: 총점 표준편차 = {grade_std:.1f}")

        print("\n✅ 조건별 성공 여부:")
        print(f"  리더십 분배: {'성공' if leader_ok else '실패'}")
        print(f"  형제자매 그룹: {'성공' if sibling_ok else '실패'}")
        print(f"  성별 균형: {'성공' if gender_ok else '실패'}")
        print(f"  피아노 균등 분배: {'성공' if piano_ok else '실패'}")
        print(f"  비등교 균등 분배: {'성공' if atrisk_ok else '실패'}")
        print(f"  운동 균등 분배: {'성공' if athletic_ok else '실패'}")
        print(f"  클럽 균등 분배: {'성공' if club_ok else '실패'}")
        print(f"  학급 인원수: {'성공' if class_size_ok else '실패'}")
        print(f"  성적 균형(표준편차 < 10): {'성공' if grade_std < 10 else '실패'}")
        print("=" * 50)
        return True
    else:
        print(f"❌ FAILED: {solver.StatusName(status)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Class assignment - Task2 modified")
    parser.add_argument('--input', required=False, default='task2.csv', help='input CSV file (default: task2.csv)')
    parser.add_argument('--output', required=False, default='assignment_task2.csv', help='output CSV file (default: assignment_task2.csv)')
    parser.add_argument('--mode', choices=['ortools'], default='ortools', help='solver mode')
    parser.add_argument('--submission-bonus', type=int, default=2, help='bonus points for submitting extra task (default: 2)')
    parser.add_argument('--transfer-weight', type=int, default=5, help='weight for assigning transfers to leader-rich classes (default: 5)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]

    cols = df.columns.tolist()
    id_col = find_col(cols, ['id','student_id','학번','ID'])
    enemies_col = find_col(cols, ['enemy','enemies','나쁜관계','안좋다','사이','hate'])
    previous_class_col = find_col(cols, ['24년 학급','24학급','전년도','작년','prev_class'])
    leader_col = find_col(cols, ['leader','leadership','Leadership','리더','리더십'])
    piano_col = find_col(cols, ['piano','Piano','피아노'])
    grade_col = find_col(cols, ['score','grade','성적','학력','점수'])
    at_risk_col = find_col(cols, ['비등교','absent','non_attend','비등교성향','등교거부','drop'])
    gender_col = find_col(cols, ['gender','sex','성별'])
    athletic_col = find_col(cols, ['운동선호','athletic','sports','운동','체육'])
    club_col = find_col(cols, ['클럽','club','부활동','동아리','활동','extracurricular'])
    transfer_col = find_col(cols, ['transfer','전학생','is_transfer'])
    sibling_col = find_col(cols, ['sibling','형제자매','형제자매ID','sibling_id','sibling id'])
    submission_col = find_col(cols, ['submission','submitted','과제제출','task_submitted','제출'])

    print("🎯 === Modified for task2.csv: sibling groups + transfer priority + submission bonus ===")
    print(f"Detected columns: ID={id_col}, enemies={enemies_col}, prev_class={previous_class_col}, leader={leader_col}, piano={piano_col}, grade={grade_col}, at_risk={at_risk_col}, gender={gender_col}, athletic={athletic_col}, club={club_col}, transfer={transfer_col}, sibling={sibling_col}, submission={submission_col}")

    n = len(df)
    id_to_idx = {}
    for idx, row in df.iterrows():
        sid = str(row[id_col]).strip() if id_col in df.columns else str(idx)
        id_to_idx[sid] = idx

    if enemies_col and enemies_col in df.columns:
        enemies_raw = df[enemies_col].apply(parse_list_field)
        enemies_idx = enemies_raw.apply(lambda x: resolve_refs(x, id_to_idx, n))
        df['enemies_idx'] = enemies_idx
        total_enemy_pairs = sum(len(enemies) for enemies in enemies_idx)
    else:
        df['enemies_idx'] = [[] for _ in range(n)]
        total_enemy_pairs = 0

    # 형제자매ID가 S1~S30 등으로 주어졌으면, 같은 S코드를 가진 학생끼리 그룹화
    sibling_groups = []
    if sibling_col and sibling_col in df.columns:
        # 형제자매ID가 비어있지 않은 학생만 추출
        sibling_id_map = {}
        for idx, val in enumerate(df[sibling_col]):
            sid = str(val).strip()
            if sid and sid.upper().startswith('S'):
                if sid not in sibling_id_map:
                    sibling_id_map[sid] = []
                sibling_id_map[sid].append(idx)
        # 2명 이상인 그룹만 sibling_groups에 추가
        sibling_groups = [group for group in sibling_id_map.values() if len(group) > 1]
        df['sibling_groups'] = [group for group in sibling_groups] + [[] for _ in range(n-len(sibling_groups))]
        total_sibling_links = sum(len(group) for group in sibling_groups)
        total_sibling_groups = len(sibling_groups)
    else:
        df['sibling_groups'] = [[] for _ in range(n)]
        total_sibling_links = 0
        total_sibling_groups = 0

    if transfer_col and transfer_col in df.columns:
        df['is_transfer'] = df[transfer_col].apply(normalize_bool)
        total_transfers = df['is_transfer'].sum()
    else:
        df['is_transfer'] = 0
        total_transfers = 0

    previous_classmates, total_previous_pairs = build_previous_classmates(df, previous_class_col)
    df['previous_classmates'] = previous_classmates

    if leader_col and leader_col in df.columns:
        df['is_leader'] = df[leader_col].apply(normalize_bool)
        total_leaders = df['is_leader'].sum()
    else:
        df['is_leader'] = 0
        total_leaders = 0

    if piano_col and piano_col in df.columns:
        df['is_piano'] = df[piano_col].apply(normalize_bool)
        total_pianos = df['is_piano'].sum()
    else:
        df['is_piano'] = 0
        total_pianos = 0

    if at_risk_col and at_risk_col in df.columns:
        df['is_at_risk'] = df[at_risk_col].apply(normalize_bool)
        total_atrisks = df['is_at_risk'].sum()
    else:
        df['is_at_risk'] = 0
        total_atrisks = 0

    if gender_col and gender_col in df.columns:
        df['gender_m'] = df[gender_col].apply(normalize_gender_male)
        total_males = df['gender_m'].sum()
        total_females = n - total_males
    else:
        df['gender_m'] = 0
        total_males = 0
        total_females = n

    if athletic_col and athletic_col in df.columns:
        df['is_athletic'] = df[athletic_col].apply(normalize_bool)
        total_athletics = df['is_athletic'].sum()
    else:
        df['is_athletic'] = 0
        total_athletics = 0

    if club_col and club_col in df.columns:
        df['is_club'] = df[club_col].apply(normalize_club)
        total_clubs = df['is_club'].sum()
    else:
        df['is_club'] = 0
        total_clubs = 0

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

    submission_bonus = args.submission_bonus
    if submission_col and submission_col in df.columns:
        df['submitted_task'] = df[submission_col].apply(normalize_bool)
        df['grade_val'] = df['grade_val'] + df['submitted_task'] * submission_bonus
        total_submitted = df['submitted_task'].sum()
    else:
        df['submitted_task'] = 0
        total_submitted = 0

    class_sizes = [100, 100, 100, 100, 100, 100]

    print(f"\n📋 데이터 요약:")
    print(f"  총 학생 수: {n}")
    print(f"  학급 구성: {class_sizes}")
    print(f"  적대관계: {total_enemy_pairs}건")
    print(f"  전년도 클래스메이트: {total_previous_pairs}건")
    print(f"  리더십 학생: {total_leaders}명")
    print(f"  전학생(우선배정) 학생 수: {total_transfers}명")
    print(f"  형제/자매 링크 수: {total_sibling_links}")
    print(f"  형제/자매 그룹 수(2명 이상): {total_sibling_groups}")
    print(f"  추가과제 제출자(가점): {total_submitted}명")
    print(f"  피아노 학생: {total_pianos}명")
    print(f"  비등교 학생: {total_atrisks}명")
    print(f"  남학생: {total_males}명, 여학생: {total_females}명")
    print(f"  운동 선호 학생: {total_athletics}명")
    print(f"  부활동/클럽 학생: {total_clubs}명")
    print(f"  평균 성적: {avg_grade:.1f}")

    success = run_ortools_final(df, args.output, class_sizes, club_col, previous_class_col, transfer_weight=args.transfer_weight)
    if not success:
        print("\n💡 실패 원인을 분석해보세요.")


if __name__ == '__main__':
    main()
