//+------------------------------------------------------------------+
//| H1Bridge.mq5 - executes the Python H1 signal file on a DEMO acct |
//| Exports closed H1 bars; opens 0.01 lot with a broker-side stop;  |
//| closes after 24 bars; refuses to trade after a drawdown kill.    |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

input string InpSymbol       = "USDJPY#";
input double InpLot          = 0.01;
input bool   InpDryRun       = true;    // true: log what would happen, send no orders
input bool   InpAllowReal    = false;   // false: refuse to run on a real account
input double InpKillPct      = 30.0;    // stop opening trades when equity < start * (1 - pct/100)
input int    InpMaxSignalAge = 900;     // seconds a signal stays valid after its bar closes
input int    InpExportBars   = 2000;
input long   InpMagic        = 424243;

CTrade   trade;
datetime g_last_bar = 0;
datetime g_last_sig_close = 0;   // bar_close of the last signal acted on (persisted in a global variable)
double   g_start_equity = 0;
bool     g_killed = false;

string BarsFile()   { return "h1_bars_" + InpSymbol + ".csv"; }
string SignalFile() { return "h1_signal_" + InpSymbol + ".json"; }
string LogFile()    { return "h1_ea_log_" + InpSymbol + ".csv"; }

void Log(string event, string detail)
  {
   int h = FileOpen(LogFile(), FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
   if(h != INVALID_HANDLE)
     {
      FileSeek(h, 0, SEEK_END);
      FileWriteString(h, TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + "," + event + "," + detail + "\n");
      FileClose(h);
     }
   PrintFormat("H1Bridge %s: %s", event, detail);
  }

string Iso(datetime t)
  {
   string s = TimeToString(t, TIME_DATE | TIME_SECONDS);
   StringReplace(s, ".", "-");
   return s;
  }

//--- export the last closed bars (shift 1 = last CLOSED bar; the forming bar is never exported)
void ExportBars()
  {
   MqlRates r[];
   ArraySetAsSeries(r, false);
   int got = CopyRates(InpSymbol, PERIOD_H1, 1, InpExportBars, r);
   if(got <= 0)
     {
      Log("export_fail", "CopyRates err=" + (string)GetLastError());
      return;
     }
   string tmp = BarsFile() + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
      return;
   FileWriteString(h, "time,open,high,low,close,volume,spread\n");
   int dg = (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS);
   for(int i = 0; i < got; i++)
      FileWriteString(h, Iso(r[i].time) + "," + DoubleToString(r[i].open, dg) + "," + DoubleToString(r[i].high, dg) + "," +
                      DoubleToString(r[i].low, dg) + "," + DoubleToString(r[i].close, dg) + "," +
                      (string)r[i].tick_volume + "," + (string)r[i].spread + "\n");
   FileClose(h);
   FileDelete(BarsFile());
   FileMove(tmp, 0, BarsFile(), 0);
  }

bool HavePosition(ulong &ticket)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) == InpSymbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
        {
         ticket = t;
         return true;
        }
     }
   return false;
  }

//--- close the position once 24 H1 bars have passed since entry
void ManageExit()
  {
   ulong tk;
   if(!HavePosition(tk))
      return;
   datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
   int bars_since = iBarShift(InpSymbol, PERIOD_H1, opened, false); // 0 = entry bar itself
   if(bars_since >= 24)
     {
      if(InpDryRun)
         Log("would_close", "ticket=" + (string)tk + " bars=" + (string)bars_since);
      else if(trade.PositionClose(tk))
         Log("closed", "ticket=" + (string)tk + " bars=" + (string)bars_since);
      else
         Log("close_fail", "ticket=" + (string)tk + " ret=" + (string)trade.ResultRetcode());
     }
  }

//--- tiny JSON field reader for our flat signal file
string Field(string js, string key)
  {
   int p = StringFind(js, "\"" + key + "\"");
   if(p < 0)
      return "";
   p = StringFind(js, ":", p) + 1;
   while(StringGetCharacter(js, p) == ' ')
      p++;
   bool quoted = (StringGetCharacter(js, p) == '"');
   if(quoted)
      p++;
   int e = p;
   while(e < StringLen(js))
     {
      ushort c = StringGetCharacter(js, e);
      if(quoted ? c == '"' : (c == ',' || c == '}'))
         break;
      e++;
     }
   return StringSubstr(js, p, e - p);
  }

void HandleSignal()
  {
   int h = FileOpen(SignalFile(), FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
   if(h == INVALID_HANDLE)
      return;
   string js = "";
   while(!FileIsEnding(h))
      js += FileReadString(h);
   FileClose(h);

   string id = Field(js, "id");
   string bc = Field(js, "bar_close");
   StringReplace(bc, "-", ".");
   datetime bar_close = StringToTime(bc);
   if(id == "" || bar_close <= 0 || bar_close <= g_last_sig_close)   // only strictly newer signals are acted on
      return;
   g_last_sig_close = bar_close;
   GlobalVariableSet("H1Bridge_last_close_" + InpSymbol, (double)bar_close);
   string side = Field(js, "side");
   double stop_dist = StringToDouble(Field(js, "stop_distance"));
   if(side != "BUY" && side != "SELL")
      return;
   if(TimeCurrent() - bar_close > InpMaxSignalAge)
     {
      Log("stale_signal", id);
      return;
     }
   ulong tk;
   if(HavePosition(tk))
     {
      Log("skip_in_position", id);
      return;
     }
   if(g_killed)
     {
      Log("skip_killed", id);
      return;
     }
   int dg = (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS);
   double ask = SymbolInfoDouble(InpSymbol, SYMBOL_ASK), bid = SymbolInfoDouble(InpSymbol, SYMBOL_BID);
   double entry = (side == "BUY") ? ask : bid;
   double sl = NormalizeDouble((side == "BUY") ? entry - stop_dist : entry + stop_dist, dg);
   if(stop_dist <= 0)
     {
      Log("bad_stop", id);
      return;
     }
   if(InpDryRun)
     {
      Log("would_open", side + " lot=" + DoubleToString(InpLot, 2) + " entry=" + DoubleToString(entry, dg) + " sl=" + DoubleToString(sl, dg) + " id=" + id);
      return;
     }
   bool ok = (side == "BUY") ? trade.Buy(InpLot, InpSymbol, 0, sl, 0, "H1Bridge " + id)
                             : trade.Sell(InpLot, InpSymbol, 0, sl, 0, "H1Bridge " + id);
   Log(ok ? "opened" : "open_fail", side + " sl=" + DoubleToString(sl, dg) + " ret=" + (string)trade.ResultRetcode() + " id=" + id);
  }

void CheckKill()
  {
   if(g_killed || g_start_equity <= 0)
      return;
   if(AccountInfoDouble(ACCOUNT_EQUITY) < g_start_equity * (1.0 - InpKillPct / 100.0))
     {
      g_killed = true;
      GlobalVariableSet("H1Bridge_killed_" + InpSymbol, 1);
      Log("KILL", "equity " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + " below limit; no new trades");
     }
  }

//--- one-off diagnostic: broker swap (rollover) terms, needed to cost multi-day holds
void LogSwaps()
  {
   string syms[] = {"USDJPY#", "USDCHF#", "EURJPY#", "GBPJPY#"};
   for(int i = 0; i < ArraySize(syms); i++)
     {
      SymbolSelect(syms[i], true);
      Log("swap", syms[i] + " mode=" + (string)SymbolInfoInteger(syms[i], SYMBOL_SWAP_MODE) +
          " long=" + DoubleToString(SymbolInfoDouble(syms[i], SYMBOL_SWAP_LONG), 4) +
          " short=" + DoubleToString(SymbolInfoDouble(syms[i], SYMBOL_SWAP_SHORT), 4) +
          " triple_day=" + (string)SymbolInfoInteger(syms[i], SYMBOL_SWAP_ROLLOVER3DAYS) +
          " point=" + DoubleToString(SymbolInfoDouble(syms[i], SYMBOL_POINT), 6) +
          " tick_value=" + DoubleToString(SymbolInfoDouble(syms[i], SYMBOL_TRADE_TICK_VALUE), 5));
     }
  }

int OnInit()
  {
   if(!InpAllowReal && AccountInfoInteger(ACCOUNT_TRADE_MODE) != ACCOUNT_TRADE_MODE_DEMO)
     {
      Print("H1Bridge: not a demo account and InpAllowReal=false -> refusing to start");
      return INIT_FAILED;
     }
   if(!SymbolSelect(InpSymbol, true))
     {
      Print("H1Bridge: symbol not found: ", InpSymbol);
      return INIT_FAILED;
     }
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetTypeFillingBySymbol(InpSymbol);   // Specification says Immediate-or-Cancel; CTrade defaults to FOK
   trade.SetDeviationInPoints(30);
   string gv = "H1Bridge_start_equity_" + InpSymbol;
   if(!GlobalVariableCheck(gv))
      GlobalVariableSet(gv, AccountInfoDouble(ACCOUNT_EQUITY));
   g_start_equity = GlobalVariableGet(gv);
   g_killed = GlobalVariableCheck("H1Bridge_killed_" + InpSymbol);
   if(GlobalVariableCheck("H1Bridge_last_close_" + InpSymbol))
      g_last_sig_close = (datetime)GlobalVariableGet("H1Bridge_last_close_" + InpSymbol);
   EventSetTimer(10);
   Log("start", StringFormat("dry=%s lot=%.2f start_equity=%.2f killed=%s", InpDryRun ? "yes" : "NO", InpLot, g_start_equity, g_killed ? "yes" : "no"));
   LogSwaps();
   ExportBars();
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTimer()
  {
   CheckKill();
   datetime cur = iTime(InpSymbol, PERIOD_H1, 0);   // open time of the forming bar
   if(cur != g_last_bar)
     {
      g_last_bar = cur;
      ExportBars();   // a new bar just opened: the previous one is closed
     }
   ManageExit();     // every tick, so a failed close is retried
   HandleSignal();
  }
