#!/usr/bin/env python3
"""
class_assignment_step3.py - 단계별 제약조건 추가

Step 3: 위험군-친구 페어링 제약조건 추가
1. 각 학생은 정확히 하나의 학급에 배정
2. 각 학급의 정확한 인원수 유지
3. 적대관계인 학생들은 같은 학급에 배정하지 않음
4. 위험군 학생은 첫 번째 친구와 같은 학급에 배정

Usage:
    python class_assignment_step3.py --input students.csv --output assignment.csv --mode ortools
"""

import argparse
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

def build_mappings(df, id_col):
    id_to_idx = {}
    for idx, row in df.iterrows():
        sid = str(row[id_col]).strip() if id_col in df.columns else str(idx)
        id_to_idx[sid] = idx
    return id_to_idx

def resolve_refs(raw_list, id_to_idx, n):
    res = []
    for token in raw_list:
        t = token.strip()
        if t == '':
            continue
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

def run_ortools_step3(df, out_path, class_sizes, time_limit=120):
    try:
        from ortools.sat.python import cp_model
    except Exception as e:
        raise RuntimeError("ortools not installed. Install with `pip install ortools`") from e

    n = len(df)
    k = len(class_sizes)
    
    print(f"학생 수: {n}, 학급 수: {k}")
    print(f"학급 크기: {class_sizes}")

    cols = df.columns.tolist()
    
    # ID 매핑 생성
    id_col = find_col(cols, ['id','student_id','학번','ID'])
    id_to_idx = build_mappings(df, id_col)

    # 적대관계 데이터 처리
    enemies_col = find_col(cols, ['enemy','enemies','나쁜관계','안좋다','사이','hate'])
    if enemies_col:
        enemies_raw = df[enemies_col].apply(parse_list_field)
        enemies_idx = enemies_raw.apply(lambda x: resolve_refs(x, id_to_idx, n))
        df['enemies_idx'] = enemies_idx
        total_enemy_pairs = sum(len(enemies) for enemies in enemies_idx)
        print(f"✅ 적대관계 {total_enemy_pairs}쌍 처리 완료")
    else:
        df['enemies_idx'] = [[] for _ in range(n)]

    # 친구관계 데이터 처리
    friends_col = find_col(cols, ['friend','friends','좋은관계','챙겨준다','buddy','friend_id'])
    if friends_col:
        friends_raw = df[friends_col].apply(parse_list_field)
        friends_idx = friends_raw.apply(lambda x: resolve_refs(x, id_to_idx, n))
        df['friends_idx'] = friends_idx
        total_friend_pairs = sum(len(friends) for friends in friends_idx)
        print(f"✅ 친구관계 {total_friend_pairs}쌍 처리 완료")
    else:
        df['friends_idx'] = [[] for _ in range(n)]

    # 위험군 데이터 처리
    at_risk_col = find_col(cols, ['비등교','absent','non_attend','비등교성향','등교거부','drop'])
    if at_risk_col:
        df['is_at_risk'] = df[at_risk_col].apply(normalize_bool)
        at_risk_count = df['is_at_risk'].sum()
        print(f"✅ 위험군 학생 {at_risk_count}명 처리 완료")
    else:
        df['is_at_risk'] = 0

    model = cp_model.CpModel()

    # Variables x[i,c] binary
    x = {}
    for i in range(n):
        for c in range(k):
            x[(i,c)] = model.NewBoolVar(f"x_{i}_{c}")

    print("✅ 변수 생성 완료")

    # 제약조건 1: 각 학생은 정확히 하나의 학급에 배정
    for i in range(n):
        model.Add(sum(x[(i,c)] for c in range(k)) == 1)
    
    print("✅ 제약조건 1 추가: 각 학생은 정확히 하나의 학급에 배정")

    # 제약조건 2: 각 학급의 정확한 인원수 유지
    for c in range(k):
        model.Add(sum(x[(i,c)] for i in range(n)) == class_sizes[c])
    
    print("✅ 제약조건 2 추가: 각 학급의 정확한 인원수 유지")

    # 제약조건 3: 적대관계 분리
    enemy_constraints = 0
    for i in range(n):
        enemies_list = df.at[i,'enemies_idx']
        if isinstance(enemies_list, list):
            for j in enemies_list:
                if j>=0 and j<n and j!=i:
                    for c in range(k):
                        model.Add(x[(i,c)] + x[(j,c)] <= 1)
                    enemy_constraints += 1
    
    print(f"✅ 제약조건 3 추가: 적대관계 분리 ({enemy_constraints}개 제약)")

    # 제약조건 4: 위험군-친구 페어링
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
    
    print(f"✅ 제약조건 4 추가: 위험군-친구 페어링 ({atrisk_friend_constraints}개 제약)")

    print("🔧 솔버 실행 중...")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"✅ 해결 성공! 상태: {solver.StatusName(status)}")
        
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
        
        print(f"📄 결과 저장: {out_path}")
        
        # 간단한 통계
        for c in range(k):
            members = out_df[out_df['assigned_class']==c]
            print(f"학급 {c}: {len(members)}명")
        
        return True
    else:
        print(f"❌ 해결 실패! 상태: {solver.StatusName(status)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Class assignment solver - Step 3 (위험군-친구 페어링 추가)")
    parser.add_argument('--input', required=True, help='input CSV with student data')
    parser.add_argument('--output', required=True, help='output CSV path')
    parser.add_argument('--mode', choices=['ortools'], default='ortools', help='solver mode')
    parser.add_argument('--time_limit', type=int, default=120, help='time limit (seconds) for OR-Tools')
    args = parser.parse_args()

    print("="*60)
    print("🎯 CLASS ASSIGNMENT - STEP 3")
    print("📋 구현된 제약조건:")
    print("   1. 각 학생은 정확히 하나의 학급에 배정")
    print("   2. 각 학급의 정확한 인원수 유지")
    print("   3. 적대관계인 학생들은 같은 학급에 배정하지 않음")
    print("   4. 위험군 학생은 첫 번째 친구와 같은 학급에 배정")
    print("="*60)

    df = pd.read_csv(args.input)
    n = len(df)
    
    # 학급 크기 설정
    if n == 200:
        class_sizes = [33,33,33,33,34,34]
    else:
        base = n // 6
        extra = n % 6
        class_sizes = [base + (1 if i < extra else 0) for i in range(6)]

    if args.mode == 'ortools':
        try:
            success = run_ortools_step3(df, args.output, class_sizes, time_limit=args.time_limit)
            if success:
                print("\n🎉 Step 3 완료! 위험군-친구 페어링 제약조건 추가 성공")
            else:
                print("\n❌ Step 3 실패")
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == '__main__':
    main()