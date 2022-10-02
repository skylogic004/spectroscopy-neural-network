from colorama import Fore, Back, Style
import colorama

def m_green(msg):
	return Fore.GREEN + msg + Fore.RESET
def m_green2(msg):
	return Back.GREEN + Fore.BLACK + msg + Style.RESET_ALL
def m_cyan(msg):
	return Fore.CYAN + msg + Fore.RESET
def m_warn1(msg):
	return Fore.YELLOW + msg + Fore.RESET
def red(msg):
	return Fore.RED + msg + Fore.RESET
def m_warn2(msg):
	return Back.YELLOW + Fore.BLACK + msg + Style.RESET_ALL
def m_user_input(msg):
	return Back.BLUE + msg + Back.RESET