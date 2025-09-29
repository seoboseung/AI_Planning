#!/usr/bin/env python3
"""
class_assignment.py

Usage:
    python class_assignment.py --input students.csv --output assignment.csv --mode ortools --time_limit 120

Modes:
  - ortools : Uses CP-SAT (requires ortools package)
  - greedy   : Deterministic greedy heuristic (no ortools required)
"""

import argparse
import math
import random
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

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
            # 소수점이 있는 경우 정수로 변환 (예: "202502.0" -> "202502")
            try:
                if '.' in p:
                    p = str(int(float(p)))
            except:
                pass  # 변환 실패 시 원본 유지
            parsed.append(p)
    return parsed

def normalize_bool(x):
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    # 명확한 True 값들
    if s in ('1','yes','y','true','t','o','예','있음','있다','leader','리더'):
        return 1
    try:
        f = float(s)
        return 1 if f>0 else 0
    except:
        # 추가 키워드 검사
        if any(k in s for k in ['yes','true','leader','리더','피아노','piano','비등교','drop','absent']):
            return 1
    return 0

def normalize_gender_male(x):
    """성별에서 남성 여부를 판단하는 전용 함수"""
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    # 남성을 나타내는 키워드들
    if s in ('male','m','boy','남','남자','남성'):
        return 1
    if any(k in s for k in ['male','boy','남자','남성']):
        return 1
    return 0

def build_mappings(df, id_col, name_col):
    id_to_idx = {}
    name_to_idx = {}
    for idx, row in df.iterrows():
        sid = str(row[id_col]).strip() if id_col in df.columns else str(idx)
        id_to_idx[sid] = idx
        if name_col and not pd.isna(row[name_col]):
            name_key = str(row[name_col]).strip().lower()
            # 중복 이름이 있을 수 있으므로 첫 번째 등장만 저장
            if name_key not in name_to_idx:
                name_to_idx[name_key] = idx
    return id_to_idx, name_to_idx# ---------- OR-Tools mode ----------
def run_ortools(df, out_path, class_sizes, time_limit=120):
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

    # Each student exactly one class
    for i in range(n):
        model.Add(sum(x[(i,c)] for c in range(k)) == 1)

    # class size constraints
    for c in range(k):
        model.Add(sum(x[(i,c)] for i in range(n)) == class_sizes[c])

    # enemies: not in same class
    enemy_constraints = 0
    for i in range(n):
        enemies_list = df.at[i,'enemies_idx']
        if isinstance(enemies_list, list):
            for j in enemies_list:
                if j>=0 and j<n and j!=i:
                    for c in range(k):
                        model.Add(x[(i,c)] + x[(j,c)] <= 1)
                    enemy_constraints += 1
    if enemy_constraints > 0:
        print(f"Added {enemy_constraints} enemy separation constraints")

    # at-risk friend constraint: if a student is at-risk and has friends, enforce same class with first friend (configurable)
    atrisk_friend_constraints = 0
    for i in range(n):
        if df.at[i,'is_at_risk']==1:
            friends_list = df.at[i,'friends_idx']
            if isinstance(friends_list, list) and len(friends_list) > 0:
                f = friends_list[0]
                if 0 <= f < n:
                    for c in range(k):
                        model.Add(x[(i,c)] == x[(f,c)])
                    atrisk_friend_constraints += 1
    if atrisk_friend_constraints > 0:
        print(f"Added {atrisk_friend_constraints} at-risk student friend pairing constraints")

    # leaders: at least one per class (if any leaders exist)
    leader_idxs = [i for i in range(n) if df.at[i,'is_leader']==1]
    if leader_idxs:
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in leader_idxs) >= 1)

    # piano balance: floor/ceil per class
    piano_idxs = [i for i in range(n) if df.at[i,'is_piano']==1]
    if piano_idxs:
        p_total = len(piano_idxs)
        p_floor = p_total // k
        p_ceil = math.ceil(p_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in piano_idxs) >= p_floor)
            model.Add(sum(x[(i,c)] for i in piano_idxs) <= p_ceil)

    # atrisk balance
    atrisk_idxs = [i for i in range(n) if df.at[i,'is_at_risk']==1]
    if atrisk_idxs:
        a_total = len(atrisk_idxs)
        a_floor = a_total // k
        a_ceil = math.ceil(a_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in atrisk_idxs) >= a_floor)
            model.Add(sum(x[(i,c)] for i in atrisk_idxs) <= a_ceil)

    # gender balance (male)
    male_idxs = [i for i in range(n) if df.at[i,'gender_m']==1]
    if male_idxs:
        m_total = len(male_idxs)
        m_floor = m_total // k
        m_ceil = math.ceil(m_total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in male_idxs) >= m_floor)
            model.Add(sum(x[(i,c)] for i in male_idxs) <= m_ceil)

    # previous class: students with same prev_class should not be in same new class
    prev_groups = defaultdict(list)
    for i in range(n):
        prev = str(df.at[i,'prev_class']).strip()
        if prev == '' or prev.lower() in ('nan','none','0'): continue
        prev_groups[prev].append(i)
    for grp in prev_groups.values():
        if len(grp) < 2: continue
        for a in range(len(grp)):
            for b in range(a+1, len(grp)):
                i = grp[a]; j = grp[b]
                for c in range(k):
                    model.Add(x[(i,c)] + x[(j,c)] <= 1)

    # club balancing (floor/ceil)
    clubs = set(sum(df['clubs'].tolist(), []))
    for club in clubs:
        members = [i for i in range(n) if club in df.at[i,'clubs']]
        total = len(members)
        if total == 0: continue
        floor_c = total // k
        ceil_c = math.ceil(total / k)
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in members) >= floor_c)
            model.Add(sum(x[(i,c)] for i in members) <= ceil_c)

    # Grade balancing objective: minimize maximum absolute deviation of class grade sums from target
    # scale grades to int
    scale = 100
    grades_int = [int(round(v * scale)) for v in df['grade_val'].tolist()]
    total_grade = sum(grades_int)
    target = total_grade // k
    class_grade_sum = []
    for c in range(k):
        s = sum(x[(i,c)] * grades_int[i] for i in range(n))
        class_grade_sum.append(s)

    # diff var bounds
    max_diff = model.NewIntVar(0, total_grade, 'max_diff')
    for c in range(k):
        model.Add(class_grade_sum[c] - target <= max_diff)
        model.Add(target - class_grade_sum[c] <= max_diff)

    model.Minimize(max_diff)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = max(1, min(8, (np.cpu_count() if hasattr(np, 'cpu_count') else 4)))
    print("Solving CP-SAT (time limit {}s)...".format(time_limit))
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Solver status:", solver.StatusName(status))
        assign = []
        for i in range(n):
            for c in range(k):
                if solver.Value(x[(i,c)]) == 1:
                    assign.append((i, c))
                    break
        out_df = df.copy()
        out_df['assigned_class'] = [c for (_, c) in sorted(assign, key=lambda x:x[0])]
        out_df.to_csv(out_path, index=False)
        print("Saved assignment to", out_path)
        # brief summary
        for c in range(k):
            members = out_df[out_df['assigned_class']==c]
            print(f"Class {c}: size {len(members)}, leaders {members['is_leader'].sum()}, piano {members['is_piano'].sum()}, atrisk {members['is_at_risk'].sum()}, male {members['gender_m'].sum()}, grade_mean {members['grade_val'].mean():.2f}")
        return True
    else:
        print("No feasible solution found. Solver status:", solver.StatusName(status))
        return False

# ---------- Greedy fallback ----------
def run_greedy(df, out_path, class_sizes, seed=42):
    random.seed(seed)
    n = len(df)
    k = len(class_sizes)
    classes = {c: [] for c in range(k)}
    # helper checks
    def violates_enemies(i,c):
        my_enemies = df.at[i,'enemies_idx'] if isinstance(df.at[i,'enemies_idx'], list) else []
        for j in classes[c]:
            # i의 적들 중에 j가 있는지 확인
            if j in my_enemies:
                return True
            # j의 적들 중에 i가 있는지 확인
            j_enemies = df.at[j,'enemies_idx'] if isinstance(df.at[j,'enemies_idx'], list) else []
            if i in j_enemies:
                return True
        return False
    def violates_prev(i,c):
        prev = str(df.at[i,'prev_class']).strip()
        if prev=='' or prev.lower() in ('nan','none','0'): return False
        for j in classes[c]:
            if str(df.at[j,'prev_class']).strip() == prev and prev!='': return True
        return False

    # 1) place leaders first (one per class if possible)
    leaders = [i for i in range(n) if df.at[i,'is_leader']==1]
    random.shuffle(leaders)
    for c in range(k):
        chosen = None
        for lid in leaders:
            if not violates_enemies(lid, c) and not violates_prev(lid, c):
                chosen = lid; break
        if chosen is None and leaders:
            chosen = leaders[0]
        if chosen is not None:
            classes[c].append(chosen)
            leaders.remove(chosen)

    # 2) pair at-risk with their first friend (if any)
    for i in range(n):
        if df.at[i,'is_at_risk']==1:
            friends_list = df.at[i,'friends_idx']
            if isinstance(friends_list, list) and len(friends_list) > 0:
                f = friends_list[0]
                if 0 <= f < n:  # 유효한 친구 인덱스인지 확인
                    # if already both assigned skip
                    assigned_i = any(i in lst for lst in classes.values())
                    assigned_f = any(f in lst for lst in classes.values())
                    if assigned_i and assigned_f: continue
                    placed=False
                    for c in range(k):
                        if len(classes[c]) + (0 if assigned_i else 1) + (0 if assigned_f else 1) > class_sizes[c]:
                            continue
                        if (not assigned_i) and (violates_enemies(i,c) or violates_prev(i,c)): continue
                        if (not assigned_f) and (violates_enemies(f,c) or violates_prev(f,c)): continue
                        if not assigned_i: classes[c].append(i)
                        if not assigned_f: classes[c].append(f)
                        placed=True; break
                    if not placed:
                        # try to place separately later
                        pass

    # 3) heuristic scoring for the rest
    unassigned = [i for i in range(n) if not any(i in lst for lst in classes.values())]
    # order by grade descending to spread high graders
    unassigned_sorted = sorted(unassigned, key=lambda i: df.at[i,'grade_val'], reverse=True)

    def score_place(i,c):
        if len(classes[c]) >= class_sizes[c]: return 1e9
        if violates_enemies(i,c): return 1e9
        if violates_prev(i,c): return 1e9
        score = 0.0
        # size pressure
        score += len(classes[c]) / class_sizes[c] * 5.0
        # piano balancing
        if df.at[i,'is_piano']==1:
            score += sum(df.at[j,'is_piano'] for j in classes[c]) * 2.0
        # atrisk balancing
        if df.at[i,'is_at_risk']==1:
            score += sum(df.at[j,'is_at_risk'] for j in classes[c]) * 2.0
        # gender balancing
        if df.at[i,'gender_m']==1:
            score += sum(df.at[j,'gender_m'] for j in classes[c]) * 1.5
        # clubs
        for club in df.at[i,'clubs']:
            score += sum(1 for j in classes[c] if club in df.at[j,'clubs']) * 1.0
        # grade balancing: prefer placing high grade into lower-sum class
        class_grade = sum(df.at[j,'grade_val'] for j in classes[c])
        avg = df['grade_val'].sum() / k
        score += (class_grade - avg) * 0.1 * (df.at[i,'grade_val'] / (df['grade_val'].max() + 1e-6))
        # friends reward
        friends_list = df.at[i,'friends_idx'] if isinstance(df.at[i,'friends_idx'], list) else []
        for f in friends_list:
            if f in classes[c]: score -= 5.0
        score += random.random()*0.01
        return score

    for i in unassigned_sorted:
        best_c, best_s = None, 1e12
        for c in range(k):
            s = score_place(i,c)
            if s < best_s:
                best_s = s; best_c = c
        if best_c is None:
            # should not happen, fallback place in first with space
            for c in range(k):
                if len(classes[c]) < class_sizes[c] and not violates_enemies(i,c) and not violates_prev(i,c):
                    best_c = c; break
        if best_c is None:
            # force place into any class with room
            for c in range(k):
                if len(classes[c]) < class_sizes[c]:
                    best_c = c; break
        classes[best_c].append(i)

    # final adjustments if any class overflow
    for c in range(k):
        while len(classes[c]) > class_sizes[c]:
            # remove non-leader first
            remove = None
            for j in classes[c]:
                if df.at[j,'is_leader']==0:
                    remove = j; break
            if remove is None:
                remove = classes[c].pop()
            else:
                classes[c].remove(remove)
                # place removed into another class
                placed=False
                for cc in range(k):
                    if len(classes[cc]) < class_sizes[cc] and not violates_enemies(remove, cc) and not violates_prev(remove, cc):
                        classes[cc].append(remove); placed=True; break
                if not placed:
                    for cc in range(k):
                        if len(classes[cc]) < class_sizes[cc]:
                            classes[cc].append(remove); break

    # ensure fill
    for c in range(k):
        if len(classes[c]) < class_sizes[c]:
            remaining = [i for i in range(n) if not any(i in lst for lst in classes.values())]
            for r in remaining:
                if len(classes[c]) < class_sizes[c] and not violates_enemies(r,c) and not violates_prev(r,c):
                    classes[c].append(r)

    # write output
    assign = []
    for c in range(k):
        for i in classes[c]:
            assign.append((i,c))
    out_df = df.copy()
    out_df['assigned_class'] = [c for (_, c) in sorted(assign, key=lambda x:x[0])]
    out_df.to_csv(out_path, index=False)
    print("Saved greedy assignment to", out_path)
    for c in range(k):
        members = out_df[out_df['assigned_class']==c]
        print(f"Class {c}: size {len(members)}, leaders {members['is_leader'].sum()}, piano {members['is_piano'].sum()}, atrisk {members['is_at_risk'].sum()}, male {members['gender_m'].sum()}, grade_mean {members['grade_val'].mean():.2f}")
    return True

# ---------- Validation function ----------
def validate_assignment(df, class_sizes):
    """
    배정 결과가 모든 제약조건을 만족하는지 검증하는 함수
    """
    print("\n" + "="*50)
    print("📋 배정 결과 검증 시작")
    print("="*50)
    
    n = len(df)
    k = len(class_sizes)
    violations = []
    warnings = []
    
    # 1. 기본 제약조건 검사
    print("\n1️⃣ 기본 제약조건 검사")
    
    # 각 학생이 정확히 하나의 학급에 배정되었는지
    assigned_classes = df['assigned_class'].tolist()
    unassigned = df[df['assigned_class'].isna()]
    if len(unassigned) > 0:
        violations.append(f"❌ {len(unassigned)}명의 학생이 배정되지 않음")
    else:
        print("✅ 모든 학생이 학급에 배정됨")
    
    # 학급 크기 확인
    for c in range(k):
        actual_size = len(df[df['assigned_class'] == c])
        expected_size = class_sizes[c]
        if actual_size != expected_size:
            violations.append(f"❌ 학급 {c}: 예상 {expected_size}명, 실제 {actual_size}명")
        else:
            print(f"✅ 학급 {c}: {actual_size}명 (목표달성)")
    
    # 2. 적대관계 제약조건 검사
    print("\n2️⃣ 적대관계 제약조건 검사")
    enemy_violations = 0
    for i in range(n):
        student_class = df.at[i, 'assigned_class']
        if pd.isna(student_class):
            continue
        student_class = int(student_class)
        
        enemies_list = df.at[i, 'enemies_idx']
        if not isinstance(enemies_list, list):
            continue
            
        for enemy_idx in enemies_list:
            if 0 <= enemy_idx < n and enemy_idx != i:
                enemy_class = df.at[enemy_idx, 'assigned_class']
                if pd.isna(enemy_class):
                    continue
                enemy_class = int(enemy_class)
                if student_class == enemy_class:
                    enemy_violations += 1
                    violations.append(f"❌ 학생 {i}와 적대관계인 학생 {enemy_idx}가 같은 학급 {student_class}에 배정됨")
    
    if enemy_violations == 0:
        print("✅ 적대관계 제약조건 모두 만족")
    else:
        print(f"❌ 적대관계 위반: {enemy_violations}건")
    
    # 3. 위험군-친구 페어링 검사
    print("\n3️⃣ 위험군-친구 페어링 검사")
    atrisk_violations = 0
    for i in range(n):
        if df.at[i, 'is_at_risk'] == 1:
            friends_list = df.at[i, 'friends_idx']
            if not isinstance(friends_list, list) or len(friends_list) == 0:
                continue
                
            student_class = df.at[i, 'assigned_class']
            if pd.isna(student_class):
                continue
            student_class = int(student_class)
            
            friend_idx = friends_list[0]
            if 0 <= friend_idx < n:
                friend_class = df.at[friend_idx, 'assigned_class']
                if pd.isna(friend_class):
                    continue
                friend_class = int(friend_class)
                if student_class != friend_class:
                    atrisk_violations += 1
                    violations.append(f"❌ 위험군 학생 {i}가 친구 {friend_idx}와 다른 학급에 배정됨 ({student_class} vs {friend_class})")
    
    if atrisk_violations == 0:
        print("✅ 위험군-친구 페어링 제약조건 모두 만족")
    else:
        print(f"❌ 위험군-친구 페어링 위반: {atrisk_violations}건")
    
    # 4. 리더 분배 검사
    print("\n4️⃣ 리더 분배 검사")
    leader_distribution = []
    for c in range(k):
        leaders_in_class = len(df[(df['assigned_class'] == c) & (df['is_leader'] == 1)])
        leader_distribution.append(leaders_in_class)
        if leaders_in_class == 0:
            violations.append(f"❌ 학급 {c}에 리더가 없음")
        else:
            print(f"✅ 학급 {c}: 리더 {leaders_in_class}명")
    
    # 5. 이전 학급 분리 검사
    print("\n5️⃣ 이전 학급 분리 검사")
    prev_class_violations = 0
    prev_groups = defaultdict(list)
    for i in range(n):
        prev = str(df.at[i, 'prev_class']).strip()
        if prev != '' and prev.lower() not in ('nan', 'none', '0'):
            prev_groups[prev].append(i)
    
    for prev_class, students in prev_groups.items():
        if len(students) < 2:
            continue
        current_classes = defaultdict(list)
        for student in students:
            current_class = df.at[student, 'assigned_class']
            if pd.isna(current_class):
                continue
            current_class = int(current_class)
            current_classes[current_class].append(student)
        
        for current_class, same_class_students in current_classes.items():
            if len(same_class_students) > 1:
                prev_class_violations += 1
                violations.append(f"❌ 이전 학급 {prev_class} 출신 학생들이 현재 학급 {current_class}에 함께 배정됨: {same_class_students}")
    
    if prev_class_violations == 0:
        print("✅ 이전 학급 분리 제약조건 모두 만족")
    else:
        print(f"❌ 이전 학급 분리 위반: {prev_class_violations}건")
    
    # 6. 균형 분배 검사
    print("\n6️⃣ 균형 분배 검사")
    
    # 피아노 균형
    piano_counts = []
    for c in range(k):
        count = len(df[(df['assigned_class'] == c) & (df['is_piano'] == 1)])
        piano_counts.append(count)
    if max(piano_counts) - min(piano_counts) <= 1:
        print(f"✅ 피아노 학생 균형: {piano_counts}")
    else:
        warnings.append(f"⚠️ 피아노 학생 불균형: {piano_counts}")
    
    # 위험군 균형
    atrisk_counts = []
    for c in range(k):
        count = len(df[(df['assigned_class'] == c) & (df['is_at_risk'] == 1)])
        atrisk_counts.append(count)
    if max(atrisk_counts) - min(atrisk_counts) <= 1:
        print(f"✅ 위험군 학생 균형: {atrisk_counts}")
    else:
        warnings.append(f"⚠️ 위험군 학생 불균형: {atrisk_counts}")
    
    # 성별 균형 (실제 데이터에서 남녀 비율이 불균형일 수 있음을 고려)
    male_counts = []
    total_males = len(df[df['gender_m'] == 1])
    for c in range(k):
        count = len(df[(df['assigned_class'] == c) & (df['gender_m'] == 1)])
        male_counts.append(count)
    
    expected_males_per_class = total_males / k
    max_deviation = max(abs(count - expected_males_per_class) for count in male_counts)
    if max_deviation <= 2:  # 클래스당 ±2명 이내면 균형잡힌 것으로 간주
        print(f"✅ 남학생 균형: {male_counts} (전체 {total_males}명)")
    else:
        warnings.append(f"⚠️ 남학생 불균형: {male_counts} (전체 {total_males}명, 예상 클래스당 {expected_males_per_class:.1f}명)")
    
    # 성적 균형
    grade_means = []
    for c in range(k):
        class_students = df[df['assigned_class'] == c]
        if len(class_students) > 0:
            mean_grade = class_students['grade_val'].mean()
            grade_means.append(round(mean_grade, 2))
        else:
            grade_means.append(0.0)
    
    grade_std = np.std(grade_means)
    if grade_std < 3.0:  # 표준편차가 3점 미만이면 균형잡힌 것으로 간주 (더 관대하게)
        print(f"✅ 학급별 평균 성적 균형: {grade_means} (표준편차: {grade_std:.2f})")
    else:
        warnings.append(f"⚠️ 학급별 평균 성적 불균형: {grade_means} (표준편차: {grade_std:.2f})")
    
    # 7. 동아리 균형 검사 (간단화)
    print("\n7️⃣ 동아리 균형 검사")
    clubs = set()
    for club_list in df['clubs']:
        clubs.update(club_list)
    
    club_imbalances = 0
    for club in clubs:
        if not club:  # 빈 문자열 제외
            continue
        club_counts = []
        for c in range(k):
            count = len(df[(df['assigned_class'] == c) & (df['clubs'].apply(lambda x: club in x))])
            club_counts.append(count)
        if max(club_counts) - min(club_counts) > 1:
            club_imbalances += 1
            warnings.append(f"⚠️ 동아리 '{club}' 불균형: {club_counts}")
    
    if club_imbalances == 0:
        print("✅ 모든 동아리가 균형있게 분배됨")
    else:
        print(f"⚠️ {club_imbalances}개 동아리에서 불균형 발견")
    
    # 8. 친구 관계 분석 (보너스)
    print("\n8️⃣ 친구 관계 분석")
    friends_together = 0
    total_friendships = 0
    for i in range(n):
        student_class = df.at[i, 'assigned_class']
        if pd.isna(student_class):
            continue
        student_class = int(student_class)
        
        friends_list = df.at[i, 'friends_idx']
        if not isinstance(friends_list, list):
            continue
            
        for friend_idx in friends_list:
            if 0 <= friend_idx < n and friend_idx != i:
                total_friendships += 1
                friend_class = df.at[friend_idx, 'assigned_class']
                if pd.isna(friend_class):
                    continue
                friend_class = int(friend_class)
                if student_class == friend_class:
                    friends_together += 1
    
    if total_friendships > 0:
        friend_ratio = friends_together / total_friendships * 100
        print(f"📊 전체 친구관계 중 {friend_ratio:.1f}% ({friends_together}/{total_friendships})가 같은 학급에 배정됨")
    
    # 최종 결과
    print("\n" + "="*50)
    print("📊 최종 검증 결과")
    print("="*50)
    
    if len(violations) == 0:
        print("🎉 모든 필수 제약조건을 만족합니다!")
    else:
        print(f"❌ {len(violations)}개의 제약조건 위반 발견:")
        for violation in violations:
            print(f"   {violation}")
    
    if len(warnings) > 0:
        print(f"\n⚠️ {len(warnings)}개의 균형 관련 주의사항:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("\n" + "="*50)
    return len(violations) == 0

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Class assignment solver (OR-Tools CP-SAT or greedy fallback)")
    parser.add_argument('--input', required=True, help='input CSV with student data')
    parser.add_argument('--output', required=True, help='output CSV path')
    parser.add_argument('--mode', choices=['ortools','greedy'], default='ortools', help='solver mode')
    parser.add_argument('--time_limit', type=int, default=120, help='time limit (seconds) for OR-Tools')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    cols = df.columns.tolist()
    # detect columns (user described potential names)
    id_col = find_col(cols, ['id','student_id','학번','ID','이름'])
    name_col = find_col(cols, ['name','이름','student_name'])
    gender_col = find_col(cols, ['gender','sex','성별'])
    grade_col = find_col(cols, ['score','grade','성적','학력','점수'])
    leader_col = find_col(cols, ['leader','leadership','리더','리더십'])
    piano_col = find_col(cols, ['piano','피아노'])
    at_risk_col = find_col(cols, ['비등교','absent','non_attend','비등교성향','등교거부','drop'])
    friends_col = find_col(cols, ['friend','friends','좋은관계','챙겨준다','buddy','friend_id'])
    enemies_col = find_col(cols, ['enemy','enemies','나쁜관계','안좋다','사이','hate'])
    prev_class_col = find_col(cols, ['prev_class','24년 학급','전년도몇반','last_class','previous_class'])
    club_col = find_col(cols, ['club','clubs','클럽','동아리','부활동','activity'])

    # create working df
    n = len(df)
    w = df.copy()
    w.index = range(n)
    # ids
    if id_col and id_col in w.columns:
        w['sid'] = w[id_col].astype(str)
    else:
        w['sid'] = w.index.astype(str)

    # booleans
    w['is_leader'] = w[leader_col].apply(normalize_bool) if leader_col in w.columns else 0
    w['is_piano'] = w[piano_col].apply(normalize_bool) if piano_col in w.columns else 0
    w['is_at_risk'] = w[at_risk_col].apply(normalize_bool) if at_risk_col in w.columns else 0

    # gender male indicator
    if gender_col and gender_col in w.columns:
        w['gender_m'] = w[gender_col].apply(normalize_gender_male)
    else:
        w['gender_m'] = 0

    # grade numeric
    if grade_col and grade_col in w.columns:
        def to_num(x):
            try:
                return float(x)
            except:
                import re
                m = re.search(r'\d+(\.\d+)?', str(x))
                return float(m.group(0)) if m else 0.0
        w['grade_val'] = w[grade_col].apply(to_num)
    else:
        w['grade_val'] = 0.0

    # clubs parse
    if club_col and club_col in w.columns:
        w['clubs'] = w[club_col].apply(parse_list_field)
    else:
        w['clubs'] = [[] for _ in range(n)]

    # friends/enemies raw parsing
    w['friends_raw'] = w[friends_col].apply(parse_list_field) if friends_col in w.columns else [[] for _ in range(n)]
    w['enemies_raw'] = w[enemies_col].apply(parse_list_field) if enemies_col in w.columns else [[] for _ in range(n)]

    # prev_class
    w['prev_class'] = w[prev_class_col].astype(str) if prev_class_col in w.columns else ''

    # build id and name maps
    id_to_idx, name_to_idx = build_mappings(w, 'sid', name_col)

    def resolve_refs(raw_list):
        res=[]
        for token in raw_list:
            t = token.strip()
            if t=='':
                continue
            # ID로 먼저 시도
            if t in id_to_idx:
                res.append(id_to_idx[t])
            # 이름으로 시도 (중복 이름이 있을 수 있으므로 첫 번째만 사용)
            elif t.lower() in name_to_idx:
                res.append(name_to_idx[t.lower()])
            else:
                # 숫자로 직접 시도
                try:
                    ii = int(t)
                    if 0 <= ii < n:
                        res.append(ii)
                except:
                    # 존재하지 않는 참조는 무시하고 경고 출력
                    print(f"⚠️ 존재하지 않는 참조 무시: {t}")
                    pass
        return res

    w['friends_idx'] = w['friends_raw'].apply(resolve_refs)
    w['enemies_idx'] = w['enemies_raw'].apply(resolve_refs)

    # class sizes default: 33x4, 34x2 (total must equal n)
    # if n != 200, distribute sizes as evenly as possible with 6 classes
    if n == 200:
        class_sizes = [33,33,33,33,34,34]
    else:
        base = n // 6
        extra = n % 6
        class_sizes = [base + (1 if i < extra else 0) for i in range(6)]

    print("Detected columns (mapping):")
    print("id_col:", id_col, "name_col:", name_col, "gender_col:", gender_col, "grade_col:", grade_col)
    print("leader_col:", leader_col, "piano_col:", piano_col, "at_risk_col:", at_risk_col)
    print("friends_col:", friends_col, "enemies_col:", enemies_col, "prev_class_col:", prev_class_col, "club_col:", club_col)
    print("Class sizes:", class_sizes)
    
    # 데이터 품질 요약
    total_friends = sum(len(friends) for friends in w['friends_idx'])
    total_enemies = sum(len(enemies) for enemies in w['enemies_idx'])
    print(f"\nData quality summary:")
    print(f"- Valid friend relationships: {total_friends}")
    print(f"- Valid enemy relationships: {total_enemies}")
    print(f"- Leaders: {w['is_leader'].sum()}")
    print(f"- Piano players: {w['is_piano'].sum()}")
    print(f"- At-risk students: {w['is_at_risk'].sum()}")
    print(f"- Male students: {w['gender_m'].sum()}")
    print(f"- Female students: {len(w) - w['gender_m'].sum()}")

    if args.mode == 'ortools':
        try:
            ok = run_ortools(w, args.output, class_sizes, time_limit=args.time_limit)
            if not ok:
                print("OR-Tools failed to find feasible solution — you can try increasing time_limit or use greedy mode.")
                return
        except Exception as e:
            print("Error running OR-Tools:", e)
            print("You can switch to --mode greedy as a fallback.")
            return
    else:
        ok = run_greedy(w, args.output, class_sizes)
        if not ok:
            return
    
    # 배정 결과 검증
    try:
        result_df = pd.read_csv(args.output)
        # assigned_class 컬럼을 정수형으로 변환
        result_df['assigned_class'] = pd.to_numeric(result_df['assigned_class'], errors='coerce')
        validate_assignment(result_df, class_sizes)
    except Exception as e:
        print(f"검증 중 오류 발생: {e}")

if __name__ == '__main__':
    main()
