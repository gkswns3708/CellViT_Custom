# AGENTS.md — AI 코드리뷰 가이드 (for “Codex”)

> 이 문서는 AI 코드리뷰 에이전트(이하 **Codex**)가 일관된 기준으로 리뷰·수정 제안을 하도록 돕는 **리뷰 규범**입니다. 팀 합의에 맞게 자유롭게 수정하세요.

---

## 0) 적용 범위

* **대상**: 애플리케이션/라이브러리 코드, 인프라(Dockerfile, IaC), 데이터/ML 파이프라인, 문서.
* **목표**: 보안·성능·신뢰성·유지보수성·테스트 커버리지 보장.
* **원칙**: *작게 변경하고, 빨리 피드백 받고, 자동화로 검증*.

---

## 1) 리뷰 원칙 (핵심)

* **맥락 먼저**: PR 설명(왜/무엇/어떻게/영향)을 기준으로 판단.
* **사실 기반**: 린트/테스트/벤치마크/로그 등 근거를 남긴다.
* **최소한의 변경**: 동일 효과라면 diff 를 작게 유지.
* **제안 지향**: 막연한 비판 대신, 적용 가능한 대안을 함께 제시.
* **자동화 우선**: 반복 피드백은 린트/테스트/CI로 이전.

---

## 2) 공통 체크리스트

### 2.1 보안(Security)

* [ ] 입력 검증: 외부 입력(파일/네트워크/사용자) 경계에서 검증/정규화.
* [ ] 인증/인가: 민감 API/리소스 접근에 권한 확인 로직 존재.
* [ ] 비밀/키: `.env`, 코드, 로그에 비밀 노출 금지. Secret manager 사용 여부.
* [ ] 암호화: 전송(TLS)·저장(예: AES-256) 필요 지점 식별/적용.
* [ ] 의존성: 신규/업데이트 패키지의 CVE 체크 및 버전 고정.
* [ ] 로깅/감사: 민감정보 마스킹, 실패·권한 거부 이벤트 로깅.
* [ ] SSRF/RCE/SQLi/XSS/N-Path: 데이터 경로 내 위험 패턴 탐지 및 차단.

### 2.2 성능(Performance)

* [ ] 복잡도: Hot path 의 Big-O 증가/감소 여부.
* [ ] 메모리/CPU: 대용량 입력에서 피크/누수 위험.
* [ ] I/O 효율: N+1 쿼리, 불필요한 동기 I/O, 과도한 직렬화/역직렬화 제거.
* [ ] 배치/스트리밍: 벡터화, 배치 처리, 제너레이터/스트리밍 활용.
* [ ] 캐싱: 재계산/재다운로드 방지 전략 존재(만료·무효화 포함).

### 2.3 신뢰성/운영(Reliability & Ops)

* [ ] 실패 처리: 재시도·백오프·타임아웃·회로차단기 적용.
* [ ] 관측성: 로그 레벨·구조화 로그·메트릭·트레이싱 추가/업데이트.
* [ ] 구성/플래그: 기능 플래그/환경변수 기본값·검증.
* [ ] 마이그레이션: 데이터 스키마·역호환·롤백 플랜.

### 2.4 유지보수성(Maintainability)

* [ ] 가독성: 함수 길이/네이밍/주석/모듈화.
* [ ] 중복/결합도: DRY, 단일책임, 인터페이스 경계 명확화.
* [ ] 타입/계약: 정적 타입 또는 런타임 검증 강화.

### 2.5 테스트 커버리지(Tests)

* [ ] 신규/변경 코드 **라인 커버리지 ≥ 80%**, 위험영역 ≥ 90%.
* [ ] 실패/경계/오류 경로 테스트 포함.
* [ ] 회귀 테스트: 발견된 버그에 대한 최소 재현 테스트 추가.

### 2.6 문서/변경관리

* [ ] README/사용 가이드/예제 동기화.
* [ ] API 변경 시 CHANGELOG와 **Deprecation** 안내.
* [ ] 릴리즈 노트: 브레이킹 체인지 명시.

---

## 3) 언어·영역별 가이드 (요약)

### Python

* 스타일: `black`, `isort`, `flake8`/`ruff` 통과.
* 성능: 루프→벡터화(`numpy`), 데이터프레임 연산은 체인 최소화.
* 타입: `mypy` 필수(핵심 경로).

### JavaScript/TypeScript

* 스타일: `eslint`, `prettier` 통과.
* 타입: `tsc --noEmit` 깨끗하게.
* 번들: 셰이크 가능 여부(Tree-shaking), 코드 스플리팅.

### Docker/Infra

* 멀티스테이지 빌드, 고정 태그(sha) 사용.
* `apt-get update && install` 후 캐시/리스트 정리.
* `HEALTHCHECK`, 최소 권한 유저.

### Data/ML

* 시드 고정·재현성(환경/버전 핀닝).
* 데이터 스키마 검증(예: `pydantic`/`pandera`).
* 메트릭 정의와 드리프트 모니터링.

---

## 4) 심각도 & 액션(라벨)

| 등급               | 기준                        | 권장 액션                 |
| ---------------- | ------------------------- | --------------------- |
| **S0 - Blocker** | 보안 취약점, 데이터 손상, 대규모 성능 퇴행 | 머지 금지, 즉시 수정 제안/패치    |
| **S1 - High**    | 장애 가능성, 주요 기능 오동작         | 머지 보류, 수정 후 재검토       |
| **S2 - Medium**  | 베스트프랙티스 위반, 커버리지 부족       | 제안 적용 권고, 필요 시 추가 테스트 |
| **S3 - Low**     | 스타일/사소한 리팩터               | 머지 허용, 추후 정리 태스크      |

---

## 5) Codex 사용 규칙

* **리뷰 트리거**: PR 코멘트에 `@codex review` (필요 시 포커스 문구 추가).
* **제안 형식**: 가능하면 GitHub **suggestion** 블록으로 제공.
* **증거 제시**: 성능은 마이크로벤치/프로파일 링크, 보안은 CWE/CVE 참고 근거.
* **존중**: 사람 리뷰어의 결정권 존중, 반복 피드백은 템플릿/CI로 이전.

예시(코드 제안 블록):

```suggestion
# 변경 제안 예시 (Python)
# TODO: 입력 검증 추가 및 early return
if not is_valid(user_input):
    return make_error("invalid input")
```

---

## 6) PR 템플릿(권장)

```markdown
## 목적
- <왜 바꾸는지 한 줄 요약>

## 변경사항
- <핵심 변경 포인트 2~4개>

## 테스트
- [ ] 유닛/통합 테스트 추가
- [ ] 재현 스텝/명령어

## 영향도
- 마이그레이션/환경변수/문서화 필요 여부
- 역호환 여부

## 리뷰 포인트
- 집중해서 봐야 할 파일/로직
```

---

# PR 코멘트 스니펫 (Codex 지시문 예시)

> 아래 문구를 PR 코멘트로 붙여 넣어 Codex에 작업을 요청하세요. 필요에 맞게 대괄호 부분을 채우면 됩니다.

## 1) 전체/집중 리뷰

* `@codex review`
* `@codex review focusing on security regressions`
* `@codex review focusing on performance of [module/function/path]`
* `@codex review only for files under [path/glob]`

## 2) 테스트 관련

* `@codex suggest unit tests for [module/function], target coverage ≥ [80]%`
* `@codex list untested branches/paths and propose tests`
* `@codex check that failure/edge cases are covered; propose minimal tests`

## 3) 성능 개선 제안

* `@codex identify hot paths and propose optimizations`
* `@codex propose a vectorized rewrite for [loop/file]`
* `@codex benchmark [function] with input size [N] and summarize`

## 4) 보안 점검

* `@codex scan for secrets, unsafe eval/exec, SQL/command injection risks`
* `@codex review authorization checks around [endpoint/resource]`
* `@codex propose safer input validation for [function]`

## 5) 스타일/품질 자동 수정

* `@codex apply lint fixes and small refactors limited to non-behavioral changes`
* `@codex convert [file] to typed code and add minimal type annotations`

## 6) 문서화/변경관리

* `@codex ensure README/CHANGELOG reflect API updates`
* `@codex generate a migration note for [breaking change]`

## 7) 위험도·릴리즈 플랜

* `@codex assess risk level (S0~S3) and list rollback steps`
* `@codex enumerate monitoring/alerting to add for this change`

---

## 팀 커스터마이징 힌트

* 심각도 기준치·커버리지 목표·지원 언어·CI 명령을 레포 현실에 맞게 조정.
* 재사용을 위해 이 파일을 `/AGENTS.md`로 두고, PR 템플릿과 함께 참조.
* 반복되는 지시문은 GitHub Saved replies(또는 PR 템플릿)로 저장.
