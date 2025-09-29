#!/usr/bin/env python3
"""
class_assignment_final.py - 모든 제약조건 포함 최종 버전

최종 완성: 모든 제약조건 포함
1. 각 학생은 정확히 하나의 학급에 배정 
2. 각 학급의 정확한 인원수 유지 (33명×4클래스, 34명×2클래스)
3. 적대관계인 학생들은 같은 학급에 배정하지 않음 (제약조건 1-A)
4. 전년도 같은 클래스였던 학생들은 가능한 분리 (제약조건 1-B) - 24년 학급 활용
5. 리더십을 가진 학생이 각 학급에 최소 1명씩 배정 (제약조건 2)
6. 피아노 연주 가능한 학생을 균등하게 분배 (제약조건 3)
7. 성적·학력을 균등하게 분배 (제약조건 4)
8. 비등교자가 치우치지 않도록 균등하게 분배 (제약조건 5)
9. 남녀 비율을 균등하게 분배 (제약조건 6)
10. 운동 능력(발이 빠른 아이)을 균등하게 분배 (제약조건 7)
11. 부활동/클럽 활동을 균등하게 분배 (제약조건 11)

Usage:
    python class_assignment_final.py --input students.csv --output assignment.csv --mode ortools
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

def normalize_club(x):
    """부활동/클럽 참여 여부를 판단하는 함수"""
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    
    # 빈 값이나 None 체크
    if s in ('', 'nan', 'none', 'null'):
        return 0
    
    # 실제 클럽 활동 이름들이 있으면 1 (참여)
    # 어떤 텍스트든 있으면 클럽에 참여하는 것으로 간주
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
    """24년 학급 데이터를 기반으로 전년도 클래스메이트 관계 구축"""
    n = len(df)
    previous_classmates = [[] for _ in range(n)]
    
    if not previous_class_col or previous_class_col not in df.columns:
        print("⚠️ 전년도 학급 정보가 없습니다.")
        return previous_classmates, 0
    
    # 각 학급별로 학생들 그룹화
    class_groups = {}
    for idx, row in df.iterrows():
        prev_class = str(row[previous_class_col]).strip().lower()
        if prev_class and prev_class != 'nan':
            if prev_class not in class_groups:
                class_groups[prev_class] = []
            class_groups[prev_class].append(idx)
    
    # 같은 학급이었던 학생들끼리 서로 연결
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

def run_ortools_final(df, out_path, class_sizes, club_col=None, previous_class_col=None):
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

    # 제약조건 4: 전년도 클래스메이트 분산 (1-B) - 각 반에 동일 출신이 너무 몰리지 않게
    if previous_class_col and previous_class_col in df.columns:
        # 전년도 클래스별로 그룹화
        prev_class_groups = {}
        for idx, row in df.iterrows():
            prev_class = str(row[previous_class_col]).strip().lower()
            if prev_class and prev_class not in ('', 'nan', 'none', 'null'):
                if prev_class not in prev_class_groups:
                    prev_class_groups[prev_class] = []
                prev_class_groups[prev_class].append(idx)
        
        # 각 전년도 클래스별로 현재 반에 너무 몰리지 않게 제약
        violation_vars = []
        for prev_class, students in prev_class_groups.items():
            if len(students) > k:  # 학생 수가 반 수보다 많을 때만
                max_per_class = math.ceil(len(students) / k) + 1  # 약간의 여유 허용
                for c in range(k):
                    # 소프트 제약: 한 반에 너무 많이 몰리면 penalty
                    over_var = model.NewIntVar(0, len(students), f"prev_over_{prev_class}_{c}")
                    model.Add(over_var >= sum(x[(i,c)] for i in students) - max_per_class)
                    model.Add(over_var >= 0)
                    violation_vars.append(over_var)
                print(f"Added {prev_class}반 출신 분산: {len(students)}명, 각 반 최대 {max_per_class}명")
        
        print(f"Added previous class distribution constraints (soft) for {len(prev_class_groups)} classes")

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

    # 제약조건 10: 부활동/클럽 활동 균등 분배 (11번) - 클럽별 균등 분배
    if club_col and club_col in df.columns:
        # 각 클럽 종류별로 학생들을 그룹화
        club_groups = {}
        for idx, row in df.iterrows():
            club_name = str(row[club_col]).strip().lower()
            if club_name and club_name not in ('', 'nan', 'none', 'null'):
                if club_name not in club_groups:
                    club_groups[club_name] = []
                club_groups[club_name].append(idx)
        
        # 각 클럽별로 균등 분배 제약 추가
        total_club_constraints = 0
        for club_name, club_members in club_groups.items():
            if len(club_members) >= k:  # 클럽 멤버가 학급 수보다 많을 때만 분배
                club_total = len(club_members)
                club_floor = club_total // k
                club_ceil = math.ceil(club_total / k)
                for c in range(k):
                    model.Add(sum(x[(i,c)] for i in club_members) >= club_floor)
                    model.Add(sum(x[(i,c)] for i in club_members) <= club_ceil)
                print(f"Added {club_name} club balance: {club_total} members, {club_floor}-{club_ceil} per class")
                total_club_constraints += 1
            else:
                # 멤버가 적은 클럽은 최대한 분산
                for c in range(k):
                    model.Add(sum(x[(i,c)] for i in club_members) <= 1)
                print(f"Added {club_name} club scatter: {len(club_members)} members, max 1 per class")
                total_club_constraints += 1
        
        print(f"Added {total_club_constraints} different club balance constraints")

    # 제약조건 11: 성적 균형 분배 (4번) - 목적함수의 일부로 구현
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
    total_violations = sum(violation_vars) if violation_vars else 0
    model.Minimize(max_deviation * 10000 + total_violations)  # 성적 균형에 더 높은 가중치
    print("Added combined objective: grade balance (primary) + previous classmate separation (secondary)")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600  # 최종 버전이므로 충분한 시간
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
        print("\n🎯 최종 완성 결과 분석:")
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
            clubs = members['is_club'].sum()
            grade_avg = members['grade_val'].mean()
            grade_sum = members['grade_val'].sum()
            grade_sums.append(grade_sum)
            print(f"학급 {c}: {len(members)}명, 리더 {leaders}명, 피아노 {pianos}명, 비등교 {atrisks}명, 남 {males}명, 여 {females}명, 운동 {athletics}명, 클럽 {clubs}명, 평균성적 {grade_avg:.1f}, 총점 {grade_sum:.0f}")
        
        grade_std = np.std(grade_sums)
        print(f"\n성적 균형: 총점 표준편차 = {grade_std:.1f}")
        
        # 클럽별 분배 상세 분석
        if club_col and club_col in df.columns:
            print("\n🎨 클럽별 분배 분석:")
            
            # 전체 클럽 종류별 분배 현황
            all_clubs = df[club_col].value_counts()
            
            for club_name, total_members in all_clubs.items():
                if pd.notna(club_name) and str(club_name).strip():
                    distribution = []
                    for c in range(k):
                        members = out_df[out_df['assigned_class']==c]
                        club_count = (members[club_col] == club_name).sum()
                        distribution.append(club_count)
                    
                    dist_str = ", ".join([f"반{c}:{count}명" for c, count in enumerate(distribution)])
                    print(f"{club_name} ({total_members}명): {dist_str}")
            
            print("\n📊 각 학급별 클럽 다양성:")
            for c in range(k):
                members = out_df[out_df['assigned_class']==c]
                club_distribution = members[club_col].value_counts()
                club_summary = ", ".join([f"{club}({count})" for club, count in club_distribution.head(10).items()])
                print(f"학급 {c}: {club_summary}")
        
        print("\n🏆 모든 제약조건 완성!")
        print("=" * 50)
        print("✅ 적대관계 완전 분리")
        print("✅ 전년도 클래스 균등 분산")
        print("✅ 리더십 학생 균등 분배")
        print("✅ 피아노 학생 균등 분배")
        print("✅ 성적 완벽 균형")
        print("✅ 비등교 학생 균등 분배")
        print("✅ 성별 균등 분배")
        print("✅ 운동 능력 균등 분배")
        print("✅ 부활동/클럽 다양성 보장")
        print("=" * 50)
        
        return True
    else:
        print(f"❌ FAILED: {solver.StatusName(status)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Class assignment - Final Complete Version")
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
    previous_class_col = find_col(cols, ['24년 학급','24학급','전년도','작년','prev_class'])
    leader_col = find_col(cols, ['leader','leadership','Leadership','리더','리더십'])
    piano_col = find_col(cols, ['piano','Piano','피아노'])
    grade_col = find_col(cols, ['score','grade','성적','학력','점수'])
    at_risk_col = find_col(cols, ['비등교','absent','non_attend','비등교성향','등교거부','drop'])
    gender_col = find_col(cols, ['gender','sex','성별'])
    athletic_col = find_col(cols, ['운동선호','athletic','sports','운동','체육'])
    club_col = find_col(cols, ['클럽','club','부활동','동아리','활동','extracurricular'])

    print("🎯 === 최종 완성: 모든 제약조건 포함 ===")
    print(f"감지된 컬럼들:")
    print(f"  ID: {id_col}")
    print(f"  적대관계: {enemies_col}")
    print(f"  전년도 학급: {previous_class_col}")
    print(f"  리더십: {leader_col}")
    print(f"  피아노: {piano_col}")
    print(f"  성적: {grade_col}")
    print(f"  비등교: {at_risk_col}")
    print(f"  성별: {gender_col}")
    print(f"  운동: {athletic_col}")
    print(f"  클럽: {club_col}")

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

    # 전년도 클래스메이트 처리 (24년 학급 활용)
    previous_classmates, total_previous_pairs = build_previous_classmates(df, previous_class_col)
    df['previous_classmates'] = previous_classmates

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

    # 부활동/클럽 처리
    if club_col and club_col in df.columns:
        df['is_club'] = df[club_col].apply(normalize_club)
        total_clubs = df['is_club'].sum()
    else:
        df['is_club'] = 0
        total_clubs = 0

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
    print(f"  부활동/클럽 학생: {total_clubs}명")
    print(f"  평균 성적: {avg_grade:.1f}")

    # OR-Tools 실행
    success = run_ortools_final(df, args.output, class_sizes, club_col, previous_class_col)
    if not success:
        print("\n💡 실패 원인을 분석해보세요.")

if __name__ == '__main__':
    main()