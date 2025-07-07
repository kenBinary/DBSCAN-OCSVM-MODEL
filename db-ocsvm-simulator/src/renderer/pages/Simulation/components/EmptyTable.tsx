export function EmptyTable() {
  return (
    <table className="w-full border border-primary-700">
      <thead>
        <tr className="text-text-light bg-table-table-header ">
          <th className="px-2.5 py-3.5">-</th>
          <th className="px-2.5 py-3.5">-</th>
          <th className="px-2.5 py-3.5">-</th>
          <th className="px-2.5 py-3.5">-</th>
          <th className="px-2.5 py-3.5">-</th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-t border-b text-[#8FB0CC]">
          <td className="text-center px-2.5 py-3.5">-</td>
          <td className="text-center px-2.5 py-3.5">-</td>
          <td className="text-center px-2.5 py-3.5">-</td>
          <td className="text-center px-2.5 py-3.5">-</td>
          <td className="text-center px-2.5 py-3.5">-</td>
        </tr>
      </tbody>
    </table>
  );
}
