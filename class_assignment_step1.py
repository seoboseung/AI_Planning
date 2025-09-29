#!/usr/bin/env python3
"""
class_assignment_step1.py - 단계별 제약조건 추가

Step 1: 가장 기본적인 제약조건만 구현
1. 각 학생은 정확히 하나의 학급에 배정
2. 각 학급의 정확한 인원수 유지

Usage:
    python class_assignment_step1.py --input students.csv --output assignment.csv --mode ortools
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

def run_ortools_step1(df, out_path, class_sizes, time_limit=120):
    try:
        from ortools.sat.python import cp_model
    except Exception as e:
        raise RuntimeError("ortools not installed. Install with `pip install ortools`") from e

    n = len(df)
    k = len(class_sizes)
    
    print(f"학생 수: {n}, 학급 수: {k}")
    print(f"학급 크기: {class_sizes}")

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
    parser = argparse.ArgumentParser(description="Class assignment solver - Step 1 (기본 제약조건만)")
    parser.add_argument('--input', required=True, help='input CSV with student data')
    parser.add_argument('--output', required=True, help='output CSV path')
    parser.add_argument('--mode', choices=['ortools'], default='ortools', help='solver mode')
    parser.add_argument('--time_limit', type=int, default=120, help='time limit (seconds) for OR-Tools')
    args = parser.parse_args()

    print("="*60)
    print("🎯 CLASS ASSIGNMENT - STEP 1")
    print("📋 구현된 제약조건:")
    print("   1. 각 학생은 정확히 하나의 학급에 배정")
    print("   2. 각 학급의 정확한 인원수 유지")
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
            success = run_ortools_step1(df, args.output, class_sizes, time_limit=args.time_limit)
            if success:
                print("\n🎉 Step 1 완료! 기본 제약조건 모두 만족")
            else:
                print("\n❌ Step 1 실패")
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == '__main__':
    main()