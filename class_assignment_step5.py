#!/usr/bin/env python3
"""
class_assignment_step5.py - 피아노 학생 균등 분배 추가

Step 5: 피아노 학생 균등 분배 추가
1. 각 학생은 정확히 하나의 학급에 배정 
2. 각 학급의 정확한 인원수 유지 (33명×4클래스, 34명×2클래스)
3. 적대관계인 학생들은 같은 학급에 배정하지 않음 (제약조건 1-A)
4. 리더십을 가진 학생이 각 학급에 최소 1명씩 배정 (제약조건 2)
5. 피아노 연주 가능한 학생을 균등하게 분배 (제약조건 3)

Usage:
    python class_assignment_step5.py --input students.csv --output assignment.csv --mode ortools
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
    if s in ('1','yes','y','true','t','o','예','있음','있다','leader','리더'):
        return 1
    try:
        f = float(s)
        return 1 if f>0 else 0
    except:
        if any(k in s for k in ['yes','true','leader','리더','피아노','piano','비등교','drop','absent']):
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

def run_ortools_step5(df, out_path, class_sizes):
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

    # 제약조건 4: 리더십 분배 (2번) - 각 학급에 최소 1명씩
    leader_idxs = [i for i in range(n) if df.at[i,'is_leader']==1]
    if leader_idxs:
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in leader_idxs) >= 1)
        print(f"Added leadership constraints: {len(leader_idxs)} leaders, min 1 per class")
    else:
        print("No leaders found - skipping leadership constraints")

    # 제약조건 5: 피아노 학생 균등 분배 (3번)
    piano_idxs = [i for i in range(n) if df.at[i,'is_piano']==1]
    if piano_idxs:
        p_total = len(piano_idxs)
        p_floor = p_total // k  # 각 학급 최소 인원
        p_ceil = math.ceil(p_total / k)  # 각 학급 최대 인원
        
        for c in range(k):
            model.Add(sum(x[(i,c)] for i in piano_idxs) >= p_floor)
            model.Add(sum(x[(i,c)] for i in piano_idxs) <= p_ceil)
        print(f"Added piano balance constraints: {p_total} piano students, {p_floor}-{p_ceil} per class")
    else:
        print("No piano students found - skipping piano constraints")

    # 목적함수: 현재는 단순히 feasible solution을 찾는 것이 목표
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
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
        for c in range(k):
            members = out_df[out_df['assigned_class']==c]
            leaders = members['is_leader'].sum()
            pianos = members['is_piano'].sum()
            print(f"학급 {c}: {len(members)}명, 리더 {leaders}명, 피아노 {pianos}명")
        
        return True
    else:
        print(f"❌ FAILED: {solver.StatusName(status)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Class assignment - Step 5 (Piano balance)")
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
    leader_col = find_col(cols, ['leader','leadership','Leadership','리더','리더십'])
    piano_col = find_col(cols, ['piano','Piano','피아노'])

    print("=== Step 5: 피아노 학생 균등 분배 추가 ===")
    print(f"감지된 컬럼들:")
    print(f"  ID: {id_col}")
    print(f"  적대관계: {enemies_col}")
    print(f"  리더십: {leader_col}")
    print(f"  피아노: {piano_col}")

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

    # 학급 크기: 33명×4클래스, 34명×2클래스
    class_sizes = [33, 33, 33, 33, 34, 34]

    print(f"\n📋 데이터 요약:")
    print(f"  총 학생 수: {n}")
    print(f"  학급 구성: {class_sizes}")
    print(f"  적대관계: {total_enemy_pairs}건")
    print(f"  리더십 학생: {total_leaders}명")
    print(f"  피아노 학생: {total_pianos}명")

    # OR-Tools 실행
    success = run_ortools_step5(df, args.output, class_sizes)
    if not success:
        print("\n💡 실패 원인을 분석해보세요.")

if __name__ == '__main__':
    main()