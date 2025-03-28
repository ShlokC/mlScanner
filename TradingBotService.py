import os
import sys
import time
import subprocess
import psutil
import win32serviceutil
import win32service
import win32event
import servicemanager

class TradingBotService(win32serviceutil.ServiceFramework):
    _svc_name_ = "TradingBotService"
    _svc_display_name_ = "Trading Bot Service"
    _svc_description_ = "Runs the trading bot and restarts it if it terminates (e.g. due to memory issues)."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        # Create an event that we will use to wait on.
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.process = None
        self.stop_requested = False

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.stop_requested = True
        win32event.SetEvent(self.hWaitStop)
        if self.process:
            try:
                self.process.terminate()
                servicemanager.LogInfoMsg("Trading bot subprocess terminated on service stop.")
            except Exception as e:
                servicemanager.LogErrorMsg(f"Error terminating subprocess: {e}")

    def SvcDoRun(self):
        servicemanager.LogInfoMsg("TradingBotService is starting.")
        self.main()

    def main(self):
        # Path to your trading bot script
        trading_script = os.path.join(os.path.dirname(sys.argv[0]), "main2.py")
        # Main loop to start and monitor the trading bot process.
        while not self.stop_requested:
            try:
                # Start the trading bot as a subprocess.
                self.process = subprocess.Popen([sys.executable, trading_script])
                servicemanager.LogInfoMsg("Started trading bot subprocess.")
                
                # Monitor the subprocess.
                while True:
                    if self.stop_requested:
                        break
                    retcode = self.process.poll()
                    if retcode is not None:
                        # Process has terminated.
                        servicemanager.LogInfoMsg(f"Trading bot terminated with code {retcode}. Restarting...")
                        break

                    # Check memory usage (e.g., restart if RSS > 500 MB)
                    try:
                        proc = psutil.Process(self.process.pid)
                        mem_rss = proc.memory_info().rss  # resident memory in bytes
                        if mem_rss > 500 * 1024 * 1024:  # 500 MB threshold
                            servicemanager.LogInfoMsg("Memory usage exceeded threshold. Restarting trading bot...")
                            self.process.terminate()
                            break
                    except Exception as mem_e:
                        servicemanager.LogErrorMsg(f"Memory monitoring error: {mem_e}")
                    
                    time.sleep(5)
            except Exception as e:
                servicemanager.LogErrorMsg(f"Error in service main loop: {e}")
            # Wait a short interval before restarting the bot.
            time.sleep(5)

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(TradingBotService)
