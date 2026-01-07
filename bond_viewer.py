import tkinter as tk
from tkinter import ttk
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import time
import socket
import sys
import ctypes
import ctypes.wintypes as wt
import configparser

# 트레이용
import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw

from tkinter import filedialog

# =========================
# API 설정
# =========================
CONFIG_FILE = "config.ini"

def load_service_key():
    cfg = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        cfg.read(CONFIG_FILE, encoding="utf-8")
        return cfg.get("data", "service_key", fallback="").strip()
    return ""

BASE_URL = "https://apis.data.go.kr/1160100/service/GetBondIssuInfoService/getBondBasiInfo"

COLUMNS = ["발행기업", "ISIN코드", "ISIN코드명", "발행일자", "만기일자", "표면이율", "모집방법"]
CACHE_FILE = "bond_cache.json"

# =========================
# Single Instance (중복 실행 방지 + 기존 창 띄우기)
# =========================
SINGLE_INSTANCE_HOST = "127.0.0.1"
SINGLE_INSTANCE_PORT = 54321
MUTEX_NAME = r"Global\BondViewer_SingleInstance_v1"

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ERROR_ALREADY_EXISTS = 183


def try_activate_existing_instance():
    """기존 인스턴스가 열어둔 소켓으로 SHOW를 보내기(있으면 창을 앞으로)."""
    try:
        with socket.create_connection((SINGLE_INSTANCE_HOST, SINGLE_INSTANCE_PORT), timeout=0.5) as s:
            s.sendall(b"SHOW")
    except Exception:
        pass


def create_single_instance_mutex_or_exit():
    """
    Tk 창 생성 전에 실행해야 함.
    이미 실행 중이면 기존 인스턴스를 띄우고 현재 프로세스 종료.
    """
    h_mutex = kernel32.CreateMutexW(None, False, wt.LPCWSTR(MUTEX_NAME))
    if not h_mutex:
        # 뭔가 이상하면(권한 등) 뮤텍스 없이 진행
        return None

    last_err = ctypes.get_last_error()
    if last_err == ERROR_ALREADY_EXISTS:
        try_activate_existing_instance()
        sys.exit(0)

    return h_mutex


def start_instance_listener(root, show_callback):
    """최초 인스턴스만 서버를 열어 SHOW 요청을 받으면 창을 띄움."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 이미 뮤텍스로 단일 인스턴스를 보장했으므로 bind는 정상이어야 함
    server.bind((SINGLE_INSTANCE_HOST, SINGLE_INSTANCE_PORT))
    server.listen(5)

    def loop():
        while True:
            try:
                conn, _addr = server.accept()
                with conn:
                    data = conn.recv(1024)
                    if data and b"SHOW" in data:
                        root.after(0, show_callback)
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True).start()
    return server


# =========================
# 표시용 날짜 포맷
# =========================
def format_yyyymmdd(val):
    """20260102 -> 2026/01/02"""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if len(s) != 8 or not s.isdigit():
        return s
    return f"{s[:4]}/{s[4:6]}/{s[6:]}"


# =========================
# 캐시 저장/로드
# =========================
def save_cache(df: pd.DataFrame, basDt: str):
    try:
        payload = {
            "basDt": basDt,
            "savedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": df.to_dict(orient="records"),
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None, None, None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        basDt = payload.get("basDt")
        savedAt = payload.get("savedAt")
        rows = payload.get("rows", [])
        df = pd.DataFrame(rows, columns=COLUMNS)
        return df, basDt, savedAt
    except Exception:
        return None, None, None


# =========================
# API 데이터 수집(재시도 + 상세 에러)
# =========================
def find_valid_basdt(max_back_days: int = 14) -> str:
    service_key = load_service_key()
    if not service_key:
        raise RuntimeError("config.ini에 service_key가 없습니다. (data 섹션)")

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for back in range(1, max_back_days + 1):
        cand = (datetime.now() - timedelta(days=back)).strftime("%Y%m%d")
        params = {
            "serviceKey": service_key,
            "numOfRows": 1,
            "pageNo": 1,
            "resultType": "json",
            "basDt": cand,
        }

        for _attempt in range(2):
            try:
                r = session.get(BASE_URL, params=params, headers=headers, timeout=25)
                if r.status_code != 200:
                    continue
                j = r.json()
                items = j.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                if isinstance(items, dict):
                    items = [items]
                if items:
                    return cand
            except Exception:
                time.sleep(0.5)

    raise RuntimeError("최근 기간에서 유효한 기준일자(basDt)를 찾지 못했습니다.")


def fetch_company_bonds(status_cb=None):
    service_key = load_service_key()
    if not service_key:
        raise RuntimeError("config.ini에 service_key가 없습니다.")

    basDt = find_valid_basdt()

    rows = []
    pageNo = 1
    numOfRows = 100

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    while True:
        if status_cb:
            status_cb(f"API 데이터 불러오는 중... (page {pageNo})")

        params = {
            "serviceKey": service_key,
            "numOfRows": numOfRows,
            "pageNo": pageNo,
            "resultType": "json",
            "basDt": basDt,
        }

        last_err = None
        items = None

        for _attempt in range(3):
            try:
                r = session.get(BASE_URL, params=params, headers=headers, timeout=60)

                if r.status_code != 200:
                    snippet = r.text[:200].replace("\n", " ")
                    raise RuntimeError(f"HTTP {r.status_code} | {snippet}")

                try:
                    j = r.json()
                except Exception:
                    snippet = r.text[:200].replace("\n", " ")
                    raise RuntimeError(f"응답이 JSON이 아님 | {snippet}")

                items = j.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                if isinstance(items, dict):
                    items = [items]

                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.7)

        if last_err is not None:
            raise RuntimeError(f"API 호출 실패(페이지 {pageNo}): {last_err}")

        if not items:
            df = pd.DataFrame(rows, columns=COLUMNS)
            return df, basDt

        for i in items:
            crno = str(i.get("crno") or "").strip()
            if not crno:
                continue

            offer = (i.get("bondOffrMcdNm") or "").strip()
            rows.append({
                "발행기업": i.get("bondIsurNm"),
                "ISIN코드": i.get("isinCd"),
                "ISIN코드명": i.get("isinCdNm"),
                "발행일자": i.get("bondIssuDt"),
                "만기일자": i.get("bondExprDt"),
                "표면이율": i.get("bondSrfcInrt"),
                "모집방법": offer,
            })

        pageNo += 1


# =========================
# 트레이 아이콘 이미지
# =========================
def create_tray_image():
    img = Image.new("RGB", (64, 64), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 54, 54), fill=(80, 160, 255))
    d.text((24, 20), "B", fill=(255, 255, 255))
    return img


class BondApp:
    def __init__(self, root):
        self.root = root
        self.root.title("회사채 발행금리 조회")
        self.root.geometry("1250x740")

        self.df = pd.DataFrame(columns=COLUMNS)
        self.filtered_df = self.df
        self._loading = False

        # -----------------------------
        # UI: 상단 검색/토글/정렬
        # -----------------------------
        top_frame = tk.Frame(root)
        top_frame.pack(fill="x", padx=10, pady=6)

        tk.Label(top_frame, text="기업명 / ISIN명 검색:").pack(side="left")
        self.search_var = tk.StringVar()
        tk.Entry(top_frame, textvariable=self.search_var, width=40).pack(side="left", padx=6)
        self.search_var.trace_add("write", self.apply_filters)

        self.only_public_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            top_frame, text="공모만 보기", variable=self.only_public_var, command=self.apply_filters
        ).pack(side="left", padx=10)

        tk.Button(top_frame, text="금리 높은 순", command=self.sort_by_rate).pack(side="left", padx=5)
        tk.Button(top_frame, text="발행일 최신 순", command=self.sort_by_date).pack(side="left", padx=5)
        tk.Button(top_frame, text="초기화", command=self.reset_view).pack(side="left", padx=5)

        tk.Button(top_frame, text="새로고침(API)", command=self.refresh_from_api).pack(side="left", padx=15)
        tk.Button(top_frame, text="CSV 내보내기", command=self.export_filtered_to_csv).pack(side="left", padx=5)
        # -----------------------------
        # 테이블
        # -----------------------------
        table_frame = tk.Frame(root)
        table_frame.pack(fill="both", expand=True, padx=10, pady=6)

        self.tree = ttk.Treeview(table_frame, columns=COLUMNS, show="headings", selectmode="extended")
        COL_WEIGHTS = {
            "발행기업": 14,
            "ISIN코드": 12,
            "ISIN코드명": 34,   # 가장 넓게
            "발행일자": 10,
            "만기일자": 10,
            "표면이율": 8,
            "모집방법": 6,
        }
        # 시작 창 너비 기준으로 초기 픽셀 폭 계산
        # (TreeView 좌우 패딩/스크롤바 여유로 약간 빼줌)
        base_width = 1250 - 60
        total_w = sum(COL_WEIGHTS.values())
        for col in COLUMNS:
            self.tree.heading(col, text=col)

            # 정렬
            if col == "표면이율":
                anchor = "e"
            else:
                anchor = "center"

            init_px = max(70, int(base_width * (COL_WEIGHTS.get(col, 10) / total_w)))

            self.tree.column(
                col,
                width=init_px,
                anchor=anchor,
                stretch=True   #  전부 stretch
            )
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # -----------------------------
        # 우클릭 컨텍스트 메뉴(선택행 ISIN 복사)
        # -----------------------------
        self._ctx_menu = tk.Menu(self.root, tearoff=0)
        self._ctx_menu.add_command(
            label="선택된 ISIN코드 복사",
            command=self._copy_selected_isin
        )

        # Windows / macOS 모두 대응
        self.tree.bind("<Button-3>", self._on_right_click)
        self.tree.bind("<Button-2>", self._on_right_click)

        # -----------------------------
        # 상태 + 프로그레스바
        # -----------------------------
        status_frame = tk.Frame(root)
        status_frame.pack(fill="x", padx=10, pady=4)

        self.status_var = tk.StringVar(value="로딩 대기 중")
        tk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")

        self.progress = ttk.Progressbar(status_frame, orient="horizontal", length=420, mode="indeterminate")
        self.progress.pack(anchor="w", pady=4)

        # -----------------------------
        # X 버튼: 종료 대신 숨김
        # -----------------------------
        self.root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)

        # -----------------------------
        # 트레이 시작
        # -----------------------------
        self.tray_icon = None
        threading.Thread(target=self._run_tray, daemon=True).start()

        # -----------------------------
        # 소켓 리스너 시작(중복 실행 시 SHOW 받기)
        # -----------------------------
        self._single_instance_server = start_instance_listener(self.root, self.show_window)

        # -----------------------------
        # 시작: 캐시 먼저 표시 → 최신 로드
        # -----------------------------
        self.load_from_cache_then_refresh()

    # ===== 트레이 =====
    def _run_tray(self):
        menu = (
            item("열기", self.show_window),
            item("새로고침(API)", self._tray_refresh),
            item("종료", self.exit_app),
        )
        self.tray_icon = pystray.Icon("bond_app", create_tray_image(), "회사채 조회", menu)
        self.tray_icon.run()

    def _tray_refresh(self, icon=None, menu_item=None):
        self.refresh_from_api()

    def hide_to_tray(self):
        self.root.withdraw()
        try:
            if self.tray_icon:
                self.tray_icon.notify("트레이로 최소화됨", "프로그램이 종료되지 않았습니다.")
        except Exception:
            pass

    def show_window(self, icon=None, menu_item=None):
        # 숨김 상태에서도 확실히 보이게
        self.root.after(0, self.root.deiconify)
        self.root.after(0, self.root.lift)
        self.root.after(0, self.root.focus_force)

    def exit_app(self, icon=None, menu_item=None):
        if self.tray_icon:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        self.root.after(0, self.root.destroy)

    # ===== 데이터 로딩 =====
    def load_from_cache_then_refresh(self):
        df, basDt, savedAt = load_cache()
        if df is not None and len(df) > 0:
            self.df = df
            self.apply_filters()
            self.status_var.set(f"캐시 표시 중 (저장: {savedAt}, 기준일자: {basDt}) | 최신 데이터 로딩 중...")
        else:
            self.status_var.set("캐시 없음 | API 데이터 로딩 중...")

        self.refresh_from_api()

    def refresh_from_api(self):
        if self._loading:
            return

        self._loading = True
        self.progress.start(12)
        self.status_var.set("API 데이터 불러오는 중...")

        threading.Thread(target=self._load_from_api_thread, daemon=True).start()

    def _load_from_api_thread(self):
        def set_status(msg: str):
            self.root.after(0, lambda: self.status_var.set(msg))

        try:
            df, basDt = fetch_company_bonds(status_cb=set_status)
            save_cache(df, basDt)

            self.df = df
            self.root.after(0, self.apply_filters)
            self.root.after(0, lambda: self.status_var.set(f"기준일자 {basDt} | 총 {len(df)}건 로드 완료"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"에러: {e}"))
        finally:
            self._loading = False
            self.root.after(0, self.progress.stop)

    # ===== 필터/표시 =====
    def apply_filters(self, *args):
        df = self.df

        if self.only_public_var.get():
            df = df[df["모집방법"].astype(str).str.strip() == "공모"]

        keyword = self.search_var.get().strip()
        if keyword:
            df = df[
                df["발행기업"].astype(str).str.contains(keyword, case=False, na=False)
                | df["ISIN코드명"].astype(str).str.contains(keyword, case=False, na=False)
            ]

        self.filtered_df = df
        self.load_table(df)

    def load_table(self, df):
        self.tree.delete(*self.tree.get_children())
        for _, row in df.iterrows():
            values = []
            for c in COLUMNS:
                v = row.get(c, "")
                if c in ("발행일자", "만기일자"):
                    v = format_yyyymmdd(v)
                values.append(v)
            self.tree.insert("", "end", values=values)

    # ===== 정렬 =====
    def sort_by_rate(self):
        df = self.filtered_df.copy()
        df["표면이율"] = pd.to_numeric(df["표면이율"], errors="coerce")
        df = df.sort_values("표면이율", ascending=False)
        self.filtered_df = df
        self.load_table(df)

    def sort_by_date(self):
        df = self.filtered_df.copy()
        df["발행일자"] = pd.to_numeric(df["발행일자"], errors="coerce")
        df = df.sort_values("발행일자", ascending=False)
        self.filtered_df = df
        self.load_table(df)

    def reset_view(self):
        self.search_var.set("")
        self.only_public_var.set(False)
        self.apply_filters()
    
    def _copy_selected_isin(self):
        """선택된 모든 행의 ISIN코드를 클립보드에 복사(줄바꿈으로)"""
        sel = self.tree.selection()
        if not sel:
            self.status_var.set("복사할 행이 선택되지 않았습니다.")
            return

        isin_idx = COLUMNS.index("ISIN코드")
        codes = []
        for iid in sel:
            vals = self.tree.item(iid, "values")
            if len(vals) > isin_idx:
                code = str(vals[isin_idx]).strip()
                if code:
                    codes.append(code)

        # 중복 제거(선택) + 순서 유지
        seen = set()
        codes_unique = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                codes_unique.append(c)

        if not codes_unique:
            self.status_var.set("선택된 행에서 ISIN코드를 찾지 못했습니다.")
            return

        text = "\n".join(codes_unique)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set(f"ISIN코드 {len(codes_unique)}개 복사 완료")

    def _on_right_click(self, event):
        """우클릭 시: 해당 행 선택 보정 후 컨텍스트 메뉴 표시"""
        row_id = self.tree.identify_row(event.y)

        # 우클릭한 위치에 행이 있으면, 선택에 포함되게 처리
        if row_id:
            current = set(self.tree.selection())
            if row_id not in current:
                # 우클릭한 행이 선택되어 있지 않으면, 그 행만 선택
                self.tree.selection_set(row_id)
                self.tree.focus(row_id)

        # 메뉴 띄우기(행이든 빈공간이든)
        try:
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._ctx_menu.grab_release()

    def export_filtered_to_csv(self):
        df = self.filtered_df.copy()

        if df is None or df.empty:
            self.status_var.set("내보낼 데이터가 없습니다.")
            return

        # 저장 파일명 기본값
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"bonds_filtered_{ts}.csv"

        path = filedialog.asksaveasfilename(
            title="CSV로 저장",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not path:
            return  # 취소

        try:
            # 날짜를 보기 좋게 포맷해서 저장(화면과 동일하게)
            for c in ("발행일자", "만기일자"):
                if c in df.columns:
                    df[c] = df[c].apply(format_yyyymmdd)

            # 엑셀 호환을 위해 utf-8-sig 추천(한글 안깨짐)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self.status_var.set(f"CSV 저장 완료: {os.path.basename(path)} ({len(df)}행)")
        except Exception as e:
            self.status_var.set(f"CSV 저장 실패: {e}")


if __name__ == "__main__":
    # Tk 만들기 전에 중복 실행 차단
    mutex_handle = create_single_instance_mutex_or_exit()

    root = tk.Tk()
    app = BondApp(root)
    root.mainloop()

    if mutex_handle:
        try:
            kernel32.CloseHandle(mutex_handle)
        except Exception:
            pass
