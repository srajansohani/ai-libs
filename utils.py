import sys
class Utils:


    def show_progress(current, total, bar_length=30):
        percent = (current / total) * 100
        filled_length = int(bar_length * current // total)
        
        bar = "#" * filled_length + "-" * (bar_length - filled_length)
        
        sys.stdout.write(f"\r[{bar}] {percent:.2f}%")
        sys.stdout.flush()

        if current == total:
            print()  # move to next line when done
        