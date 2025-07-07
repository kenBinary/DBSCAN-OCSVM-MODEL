import logo from "../assets/icons/logo.png";

export function NavigationHeader() {
  return (
    <nav className="flex justify-between p-4 border-b-2 border-text-light bg-background-dark text-text-light text-lg">
      <div className="flex flex-row gap-4 items-center">
        <img src={logo} alt="logo" className="w-5 h-5" />
        <p className="text-lg font-bold">
          DB-OCSVM Network Intrusion Detection
        </p>
      </div>
    </nav>
  );
}
