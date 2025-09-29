#!/usr/bin/env python3
"""
class_assignment_step2.py - 단계별 제약조건 추가

Step 2: 적대관계 제약조건 추가
1. 각 학생은 정확히 하나의 학급에 배정
2. 각 학급의 정확한 인원수 유지
3. 적대관계인 학생들은 같은 학급에 배정하지 않음

Usage:
    python class_assignment_step2.py --input students.csv --output assignment.csv --mode ortools
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

def run_ortools_step2(df, out_path, class_sizes, time_limit=120):
    try:
        from ortools.sat.python import cp_model
    except Exception as e:
        raise RuntimeError("ortools not installed. Install with `pip install ortools`") from e

    n = len(df)
    k = len(class_sizes)
    
    print(f"학생 수: {n}, 학급 수: {k}")
    print(f"학급 크기: {class_sizes}")

    # 적대관계 데이터 처리
    cols = df.columns.tolist()
    enemies_col = find_col(cols, ['enemy','enemies','나쁜관계','안좋다','사이','hate'])
    
    if enemies_col:
        print(f"✅ 적대관계 컬럼 발견: {enemies_col}")
        
        # ID 매핑 생성
        id_col = find_col(cols, ['id','student_id','학번','ID'])
        id_to_idx = build_mappings(df, id_col)
        
        # 적대관계 파싱
        enemies_raw = df[enemies_col].apply(parse_list_field)
        enemies_idx = enemies_raw.apply(lambda x: resolve_refs(x, id_to_idx, n))
        df['enemies_idx'] = enemies_idx
        
        total_enemy_pairs = sum(len(enemies) for enemies in enemies_idx)
        print(f"✅ 적대관계 {total_enemy_pairs}쌍 처리 완료")
    else:
        print("⚠️ 적대관계 컬럼을 찾을 수 없음")
        df['enemies_idx'] = [[] for _ in range(n)]

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
    parser = argparse.ArgumentParser(description="Class assignment solver - Step 2 (적대관계 추가)")
    parser.add_argument('--input', required=True, help='input CSV with student data')
    parser.add_argument('--output', required=True, help='output CSV path')
    parser.add_argument('--mode', choices=['ortools'], default='ortools', help='solver mode')
    parser.add_argument('--time_limit', type=int, default=120, help='time limit (seconds) for OR-Tools')
    args = parser.parse_args()

    print("="*60)
    print("🎯 CLASS ASSIGNMENT - STEP 2")
    print("📋 구현된 제약조건:")
    print("   1. 각 학생은 정확히 하나의 학급에 배정")
    print("   2. 각 학급의 정확한 인원수 유지")
    print("   3. 적대관계인 학생들은 같은 학급에 배정하지 않음")
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
            success = run_ortools_step2(df, args.output, class_sizes, time_limit=args.time_limit)
            if success:
                print("\n🎉 Step 2 완료! 적대관계 제약조건 추가 성공")
            else:
                print("\n❌ Step 2 실패")
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == '__main__':
    main()